"""
File containing CamstimEphysSession class
"""

import argparse
import datetime
import io
import json
from pathlib import Path

import aind_data_schema
import aind_data_schema.components.coordinates
import aind_data_schema.core.session as session_schema
import aind_data_schema_models.modalities
import np_session
import npc_ephys
import npc_mvr
import npc_sessions
import npc_sync
import numpy as np
import pandas as pd
import re
from utils import pickle_functions as pkl_utils

import aind_metadata_mapper.stimulus.camstim


class CamstimEphysSession(aind_metadata_mapper.stimulus.camstim.Camstim):
    """
    An Ephys session, designed for OpenScope, employing neuropixel
    probes with visual and optogenetic stimulus from Camstim.
    """

    json_settings: dict = None
    npexp_path: Path
    recording_dir: Path

    def __init__(self, session_id: str, json_settings: dict) -> None:
        """
        Determine needed input filepaths from np-exp and lims, get session
        start and end times from sync file, and extract epochs from stim
        tables.
        """
        self.json_settings = json_settings
        session_inst = np_session.Session(session_id)
        self.mtrain = session_inst.mtrain
        self.npexp_path = session_inst.npexp_path
        self.folder = session_inst.folder
        # sometimes data files are deleted on npexp so try files on lims
        try:
            self.recording_dir = npc_ephys.get_single_oebin_path(
                session_inst.lims_path
            ).parent
        except FileNotFoundError:
            self.recording_dir = npc_ephys.get_single_oebin_path(
                session_inst.npexp_path
            ).parent

        self.motor_locs_path = (
            self.npexp_path / f"{self.folder}.motor-locs.csv"
        )
        self.pkl_path = self.npexp_path / f"{self.folder}.stim.pkl"
        self.opto_table_path = (
            self.npexp_path / f"{self.folder}_opto_epochs.csv"
        )
        self.stim_table_path = (
            self.npexp_path / f"{self.folder}_stim_epochs.csv"
        )
        self.sync_path = self.npexp_path / f"{self.folder}.sync"

        platform_path = next(
            self.npexp_path.glob(f"{self.folder}_platform*.json")
        )
        self.platform_json = json.loads(platform_path.read_text())
        self.project_name = self.platform_json["project"]

        sync_data = npc_sync.SyncDataset(
            io.BytesIO(self.sync_path.read_bytes())
        )
        self.session_start, self.session_end = (
            sync_data.start_time,
            sync_data.stop_time,
        )
        print("session start:end", self.session_start, ":", self.session_end)

        print("getting stim epochs")
        self.stim_epochs = self.epochs_from_stim_table()

        if self.opto_table_path.exists():
            self.stim_epochs.append(self.epoch_from_opto_table())

        self.available_probes = self.get_available_probes()

    def generate_session_json(self) -> session_schema.Session:
        """
        Creates the session schema json
        """
        self.session_json = session_schema.Session(
            experimenter_full_name=[
                self.platform_json["operatorID"].replace(".", " ").title()
            ],
            session_start_time=self.session_start,
            session_end_time=self.session_end,
            session_type=self.json_settings.get("session_type", ""),
            iacuc_protocol=self.json_settings.get("iacuc_protocol", ""),
            rig_id=self.platform_json["rig_id"],
            subject_id=self.folder.split("_")[1],
            data_streams=self.data_streams(),
            stimulus_epochs=self.stim_epochs,
            mouse_platform_name=self.json_settings.get(
                "mouse_platform", "Mouse Platform"
            ),
            active_mouse_platform=self.json_settings.get(
                "active_mouse_platform", False
            ),
            reward_consumed_unit="milliliter",
            notes="",
        )
        return self.session_json

    def write_session_json(self) -> None:
        """
        Writes the session json to a session.json file
        """
        self.session_json.write_standard_file(self.npexp_path)
        print(f"File created at {str(self.npexp_path)}/session.json")

    def get_available_probes(self) -> tuple[str]:
        """
        Returns a list of probe letters among ABCDEF that are inserted
        according to platform.json. If platform.json has no insertion record,
        returns all probes (this could cause problems).
        """
        insertion_notes = self.platform_json["InsertionNotes"]
        if insertion_notes == {}:
            available_probes = "ABCDEF"
        else:
            available_probes = [
                letter
                for letter in "ABCDEF"
                if not insertion_notes.get(f"Probe{letter}", {}).get(
                    "FailedToInsert", False
                )
            ]
        print("available probes:", available_probes)
        return tuple(available_probes)

    def manipulator_coords(
        self, probe_name: str, newscale_coords: pd.DataFrame
    ) -> tuple[aind_data_schema.components.coordinates.Coordinates3d, str]:
        """
        Returns the schema coordinates object containing probe's manipulator
        coordinates accrdong to newscale, and associated 'notes'. If the
        newscale coords don't include this probe (shouldn't happen), return
        coords with 0.0s and notes indicating no coordinate info available
        """
        try:
            probe_row = newscale_coords.query(
                f"electrode_group == '{probe_name}'"
            )
        except pd.errors.UndefinedVariableError:
            probe_row = newscale_coords.query(
                f"electrode_group_name == '{probe_name}'"
            )
        if probe_row.empty:
            return (
                aind_data_schema.components.coordinates.Coordinates3d(
                    x="0.0", y="0.0", z="0.0", unit="micrometer"
                ),
                "Coordinate info not available",
            )
        else:
            x, y, z = (
                probe_row["x"].item(),
                probe_row["y"].item(),
                probe_row["z"].item(),
            )
        return (
            aind_data_schema.components.coordinates.Coordinates3d(
                x=x, y=y, z=z, unit="micrometer"
            ),
            "",
        )

    def ephys_modules(self) -> list:
        """
        Return list of schema ephys modules for each available probe.
        """
        newscale_coords = npc_sessions.get_newscale_coordinates(
            self.motor_locs_path
        )

        ephys_modules = []
        for probe_letter in self.available_probes:
            probe_name = f"probe{probe_letter}"
            manipulator_coordinates, notes = self.manipulator_coords(
                probe_name, newscale_coords
            )

            probe_module = session_schema.EphysModule(
                assembly_name=probe_name.upper(),
                arc_angle=0.0,
                module_angle=0.0,
                rotation_angle=0.0,
                primary_targeted_structure="none",
                ephys_probes=[
                    session_schema.EphysProbeConfig(name=probe_name.upper())
                ],
                manipulator_coordinates=manipulator_coordinates,
                notes=notes,
            )
            ephys_modules.append(probe_module)
        return ephys_modules

    def ephys_stream(self) -> session_schema.Stream:
        """
        Returns schema ephys datastream, including the list of ephys modules
        and the ephys start and end times.
        """
        modality = aind_data_schema_models.modalities.Modality

        probe_exp = r"(?<=[pP{1}]robe)[-_\s]*(?P<letter>[A-F]{1})(?![a-zA-Z])"
        def extract_probe_letter(s):
            match = re.search(probe_exp, s)
            if match:
                return match.group("letter")

        times = npc_ephys.get_ephys_timing_on_sync(
            sync=self.sync_path, recording_dirs=[self.recording_dir]
        )

        ephys_timing_data = tuple(
            timing
            for timing in times
            if (p := extract_probe_letter(timing.device.name))
            is None
            or p in self.available_probes
        )

        stream_first_time = min(
            timing.start_time for timing in ephys_timing_data
        )
        stream_last_time = max(
            timing.stop_time for timing in ephys_timing_data
        )

        return session_schema.Stream(
            stream_start_time=self.session_start
            + datetime.timedelta(seconds=stream_first_time),
            stream_end_time=self.session_start
            + datetime.timedelta(seconds=stream_last_time),
            ephys_modules=self.ephys_modules(),
            stick_microscopes=[],
            stream_modalities=[modality.ECEPHYS],
        )

    def sync_stream(self) -> session_schema.Stream:
        """
        Returns schema behavior stream for the sync timing.
        """
        modality = aind_data_schema_models.modalities.Modality
        return session_schema.Stream(
            stream_start_time=self.session_start,
            stream_end_time=self.session_end,
            stream_modalities=[modality.BEHAVIOR],
            daq_names=["Sync"],
        )

    def video_stream(self) -> session_schema.Stream:
        """
        Returns schema behavior videos stream for video timing
        """
        modality = aind_data_schema_models.modalities.Modality
        video_frame_times = npc_mvr.mvr.get_video_frame_times(
            self.sync_path, self.npexp_path
        )

        stream_first_time = min(
            np.nanmin(timestamps) for timestamps in video_frame_times.values()
        )
        stream_last_time = max(
            np.nanmax(timestamps) for timestamps in video_frame_times.values()
        )

        return session_schema.Stream(
            stream_start_time=self.session_start
            + datetime.timedelta(seconds=stream_first_time),
            stream_end_time=self.session_start
            + datetime.timedelta(seconds=stream_last_time),
            camera_names=["Front camera", "Side camera", "Eye camera"],
            stream_modalities=[modality.BEHAVIOR_VIDEOS],
        )

    def data_streams(self) -> tuple[session_schema.Stream, ...]:
        """
        Return three schema datastreams; ephys, behavior, and behavior videos.
        May be extended.
        """
        data_streams = []
        data_streams.append(self.ephys_stream())
        data_streams.append(self.sync_stream())
        data_streams.append(self.video_stream())
        return tuple(data_streams)


def parse_args() -> argparse.Namespace:
    """
    Parse Arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate a session.json file for an ephys session"
    )
    parser.add_argument(
        "session_id",
        help=(
            "session ID (lims or np-exp foldername) or path to session"
            "folder"
        ),
    )
    parser.add_argument(
        "json-settings",
        help=(
            'json containing at minimum the fields "session_type" and'
            '"iacuc protocol"'
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Run Main
    """
    sessionETL = CamstimEphysSession(**vars(parse_args()))
    sessionETL.generate_session_json()


if __name__ == "__main__":
    main()
