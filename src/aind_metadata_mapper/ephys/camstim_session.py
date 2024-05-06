import aind_data_schema
import aind_data_schema.core.session as session_schema
import argparse
import datetime
import io
import json
import npc_ephys
import npc_mvr
import np_session
import npc_session
import npc_sessions
import npc_sync
import numpy as np
import pandas as pd
from aind_data_schema.models.modalities import Modality as SchemaModality
from aind_data_schema.models.coordinates import Coordinates3d as SchemaCoordinates
from pathlib import Path
from utils import process_ephys_sync as stim_utils
from utils import pickle_functions as pkl_utils


# defaults
DEFAULT_OPTO_CONDITIONS = {
    "0": {
        "duration": .01,
        "name": "1Hz_10ms",
        "condition": "10 ms pulse at 1 Hz"
    },
    "1": {
        "duration": .002,
        "name": "1Hz_2ms",
        "condition": "2 ms pulse at 1 Hz"
    },
    "2": {
        "duration": 1.0,
        "name": "5Hz_2ms",
        "condition": "2 ms pulses at 5 Hz"
    },
    "3": {
        "duration": 1.0,
        "name": "10Hz_2ms",
        "condition": "2 ms pulses at 10 Hz'"
    },
    "4": {
        "duration": 1.0,
        "name": "20Hz_2ms",
        "condition": "2 ms pulses at 20 Hz"
    },
    "5": {
        "duration": 1.0,
        "name": "30Hz_2ms",
        "condition": "2 ms pulses at 30 Hz"
    },
    "6": {
        "duration": 1.0,
        "name": "40Hz_2ms",
        "condition": "2 ms pulses at 40 Hz"
    },
    "7": {
        "duration": 1.0,
        "name": "50Hz_2ms",
        "condition": "2 ms pulses at 50 Hz"
    },
    "8": {
        "duration": 1.0,
        "name": "60Hz_2ms",
        "condition": "2 ms pulses at 60 Hz"
    },
    "9": {
        "duration": 1.0,
        "name": "80Hz_2ms",
        "condition": "2 ms pulses at 80 Hz"
    },
    "10": {
        "duration": 1.0,
        "name": "square_1s",
        "condition": "1 second square pulse: continuously on for 1s"
    },
    "11": {
        "duration": 1.0,
        "name": "cosine_1s",
        "condition": "cosine pulse"
    },
}


class CamstimSession():
    json_settings: dict = None
    npexp_path: Path
    recording_dir: Path


    def __init__(self, session_id: str, json_settings: dict) -> None:
        self.json_settings = json_settings
        session_inst = np_session.Session(session_id)
        self.mtrain = session_inst.mtrain
        self.npexp_path = session_inst.npexp_path
        self.folder = session_inst.folder
        # sometimes data files are deleted on npexp, better to try files on lims
        try:
            self.recording_dir = npc_ephys.get_single_oebin_path(session_inst.lims_path).parent
        except:
            self.recording_dir = npc_ephys.get_single_oebin_path(session_inst.npexp_path).parent

        self.motor_locs_path = self.npexp_path / f'{self.folder}.motor-locs.csv'
        self.pkl_path = self.npexp_path / f'{self.folder}.stim.pkl'
        self.opto_pkl_path = self.npexp_path / f'{self.folder}.opto.pkl'
        self.opto_table_path = self.npexp_path / f'{self.folder}_opto_epochs.csv' 
        self.stim_table_path = self.npexp_path / f'{self.folder}_stim_epochs.csv' 
        self.sync_path = self.npexp_path / f'{self.folder}.sync'

        platform_path = next(self.npexp_path.glob(f'{self.folder}_platform*.json'))
        self.platform_json = json.loads(platform_path.read_text())
        self.project_name = self.platform_json['project']

        sync_data = npc_sync.SyncDataset(io.BytesIO(self.sync_path.read_bytes()))
        self.session_start, self.session_end  = sync_data.start_time, sync_data.stop_time
        print('session start:end', self.session_start, ':', self.session_end)

        print("getting stim epochs")
        self.stim_epochs = self.epochs_from_stim_table()

        if self.opto_pkl_path.exists() and not self.opto_table_path.exists():
            opto_conditions = self.experiment_info[self.project_name].get('opto_conditions', DEFAULT_OPTO_CONDITIONS)
            stim_utils.build_optogenetics_table(self.opto_pkl_path, self.sync_path, opto_conditions, self.opto_table_path)
        if self.opto_table_path.exists():
            self.stim_epochs.append(self.epoch_from_opto_table())

        self.available_probes = self.get_available_probes()


    def generate_session_json(self) -> None:
        """
        Creates the session.json file
        """
        session_json = session_schema.Session(
            experimenter_full_name=[self.platform_json['operatorID'].replace('.', ' ').title()],
            session_start_time=self.session_start,
            session_end_time=self.session_end,
            session_type=self.json_settings.get('session_type', ''),
            iacuc_protocol=self.json_settings.get('iacuc_protocol',''),
            rig_id=self.platform_json['rig_id'],
            subject_id=self.folder.split('_')[1],
            data_streams=self.data_streams(),
            stimulus_epochs=self.stim_epochs,
            mouse_platform_name=self.json_settings.get('mouse_platform','Mouse Platform'),
            active_mouse_platform=self.json_settings.get('active_mouse_platform', False),
            reward_consumed_unit='milliliter',
            notes='',
        )
        session_json.write_standard_file(self.npexp_path)
        print(f'File created at {str(self.npexp_path)}/session.json')


    def get_available_probes(self) -> tuple[str]:
        """
        Returns a list of probe letters among ABCDEF that are inserted according to platform.json
        If platform.json has no insertion record, returns all probes (this could cause problems).
        """
        insertion_notes = self.platform_json['InsertionNotes']
        if insertion_notes == {}:
            available_probes = 'ABCDEF'
        else:
            available_probes = [letter for letter in 'ABCDEF' if not insertion_notes.get(f'Probe{letter}', {}).get('FailedToInsert', False)]
        print('available probes:',available_probes)
        return tuple(available_probes)


    def manipulator_coords(self, probe_name: str, newscale_coords: pd.DataFrame) -> tuple[SchemaCoordinates, str]:
        """
        Returns the schema coordinates object containing probe's manipulator coordinates accrdong to newscale, and associated 'notes'
        If the newscale coords don't include this probe (shouldn't happen), return coords with 0.0s and notes indicating no coordinate info available
        """
        probe_row = newscale_coords.query(f"electrode_group == '{probe_name}'")
        if probe_row.empty:
            return SchemaCoordinates(x='0.0', y='0.0', z='0.0', unit='micrometer'), 'Coordinate info not available'
        else:
            x, y, z = probe_row['x'].item(), probe_row['y'].item(), probe_row['z'].item()
        return SchemaCoordinates(x=x, y=y, z=z, unit='micrometer'), ''


    def ephys_modules(self) -> session_schema.EphysModule:
        """
        Return list of schema ephys modules for each available probe.
        """
        newscale_coords = npc_sessions.get_newscale_coordinates(self.motor_locs_path)
        print(newscale_coords)

        ephys_modules = []
        for probe_letter in self.available_probes:
            probe_name = f'probe{probe_letter}'
            manipulator_coordinates, notes = self.manipulator_coords(probe_name, newscale_coords)

            probe_module = session_schema.EphysModule(
                assembly_name=probe_name.upper(),
                arc_angle=0.0,
                module_angle=0.0,
                rotation_angle=0.0,
                primary_targeted_structure='none',
                ephys_probes=[session_schema.EphysProbeConfig(name=probe_name.upper())],
                manipulator_coordinates=manipulator_coordinates,
                notes=notes
            )
            ephys_modules.append(probe_module)
        return ephys_modules


    def ephys_stream(self) -> session_schema.Stream:
        """
        Returns schema ephys datastream, including the list of ephys modules and the ephys start and end times.
        """
        times = npc_ephys.get_ephys_timing_on_sync(sync=self.sync_path, recording_dirs=[self.recording_dir])
        ephys_timing_data = tuple(
            timing for timing in times if \
                (p := npc_session.extract_probe_letter(timing.device.name)) is None or p in self.available_probes
        )

        stream_first_time = min(timing.start_time for timing in ephys_timing_data)
        stream_last_time = max(timing.stop_time for timing in ephys_timing_data)

        return session_schema.Stream(
            stream_start_time=self.session_start + datetime.timedelta(seconds=stream_first_time),
            stream_end_time=self.session_start + datetime.timedelta(seconds=stream_last_time),
            ephys_modules=self.ephys_modules(),
            stick_microscopes=[],
            stream_modalities=[SchemaModality.ECEPHYS]
        )


    def sync_stream(self) -> session_schema.Stream:
        """
        Returns schema behavior stream for the sync timing.
        """
        return session_schema.Stream(
                stream_start_time=self.session_start,
                stream_end_time=self.session_end,
                stream_modalities=[SchemaModality.BEHAVIOR],
                daq_names=['Sync']
        )


    def video_stream(self) -> session_schema.Stream:
        """
        Returns schema behavior videos stream for video timing
        """
        video_frame_times = npc_mvr.mvr.get_video_frame_times(self.sync_path, self.npexp_path)

        stream_first_time = min(np.nanmin(timestamps) for timestamps in video_frame_times.values())
        stream_last_time = max(np.nanmax(timestamps) for timestamps in video_frame_times.values())

        return session_schema.Stream(
            stream_start_time=self.session_start + datetime.timedelta(seconds=stream_first_time),
            stream_end_time=self.session_start + datetime.timedelta(seconds=stream_last_time),
            camera_names=['Front camera', 'Side camera', 'Eye camera'],
            stream_modalities=[SchemaModality.BEHAVIOR_VIDEOS],
        )


    def data_streams(self) -> tuple[session_schema.Stream, ...]:
        """
        Return three schema datastreams; ephys, behavior, and behavior videos. May be extended.
        """
        data_streams = []
        data_streams.append(self.ephys_stream())
        data_streams.append(self.sync_stream())
        data_streams.append(self.video_stream())
        return tuple(data_streams)


    def epoch_from_opto_table(self) -> session_schema.StimulusEpoch:
        """
        From the optogenetic stimulation table, returns a single schema stimulus epoch representing the optotagging period.
        Include all unknown table columns (not start_time, stop_time, stim_name) as parameters, and include the set of all
        of that column's values as the parameter values.
        """
        stim = aind_data_schema.core.session.StimulusModality

        script_obj = aind_data_schema.models.devices.Software(
            name=self.mtrain['regimen']['name'],
            version='1.0',
            url=self.mtrain['regimen']['script']
        )

        opto_table = pd.read_csv(self.opto_table_path)

        opto_params = {}
        for column in opto_table:
            if column in ('start_time', 'stop_time', 'stim_name'):
                continue
            param_set = set(opto_table[column].dropna())
            opto_params[column] = param_set

        params_obj = session_schema.VisualStimulation(
            stimulus_name="Optogenetic Stimulation",
            stimulus_parameters=opto_params,
            stimulus_template_name=[]
        )

        opto_epoch = session_schema.StimulusEpoch(
            stimulus_start_time=self.session_start + datetime.timedelta(seconds=opto_table.start_time.iloc[0]),
            stimulus_end_time=self.session_start + datetime.timedelta(seconds=opto_table.start_time.iloc[-1]),
            stimulus_name="Optogenetic Stimulation",
            software=[],
            script=script_obj,
            stimulus_modalities=[stim.OPTOGENETICS],
            stimulus_parameters=[params_obj],
        )

        return opto_epoch


    def extract_stim_epochs(self, stim_table: pd.DataFrame) -> list[list[str, int, int, dict, set]]:
        """
        Returns a list of stimulus epochs, where an epoch takes the form (name, start, stop, params_dict, template names).
        Iterates over the stimulus epochs table, identifying epochs based on when the 'stim_name' field of the table changes.
        
        For each epoch, every unknown column (not start_time, stop_time, stim_name, stim_type, or frame) are listed as parameters,
        and the set of values for that column are listed as parameter values.        
        """
        epochs = []

        current_epoch = [None, 0.0, 0.0, {}, set()]
        epoch_start_idx = 0
        for current_idx, row in stim_table.iterrows():
            # if the stim name changes, summarize current epoch's parameters and start a new epoch
            if row['stim_name'] != current_epoch[0]:
                for column in stim_table:
                    if column not in ('start_time', 'stop_time', 'stim_name', 'stim_type', 'frame'):
                        param_set = set(stim_table[column][epoch_start_idx:current_idx].dropna())
                        current_epoch[3][column] = param_set

                epochs.append(current_epoch)
                epoch_start_idx = current_idx
                current_epoch = [row['stim_name'], row['start_time'], row['stop_time'], {}, set()]
            # if stim name hasn't changed, we are in the same epoch, keep pushing the stop time
            else:
                current_epoch[2] = row['stop_time']

            # if this row is a movie or image set, record it's stim name in the epoch's templates entry
            if 'image' in row.get('stim_type','').lower() or 'movie' in row.get('stim_type','').lower():
                current_epoch[4].add(row['stim_name'])

        # slice off dummy epoch from beginning
        return epochs[1:]


    def epochs_from_stim_table(self) -> list[session_schema.StimulusEpoch]:
        """
        From the stimulus epochs table, return a list of schema stimulus epochs representing the various periods of stimulus from the session.
        Also include the camstim version from pickle file and stimulus script used from mtrain.
        """
        stim = aind_data_schema.core.session.StimulusModality

        software_obj = aind_data_schema.models.devices.Software(
            name='camstim',
            version=pkl_utils.load_pkl(self.pkl_path)['platform']['camstim'].split('+')[0],
            url='https://eng-gitlab.corp.alleninstitute.org/braintv/camstim'
        )

        script_obj = aind_data_schema.models.devices.Software(
            name=self.mtrain['regimen']['name'],
            version='1.0',
            url=self.mtrain['regimen']['script']
        )

        schema_epochs = []
        for epoch_name, epoch_start, epoch_end, stim_params, stim_template_names in self.extract_stim_epochs(pd.read_csv(self.stim_table_path)):
            params_obj = session_schema.VisualStimulation(
                stimulus_name=epoch_name,
                stimulus_parameters=stim_params,
                stimulus_template_name=stim_template_names
            )

            epoch_obj = session_schema.StimulusEpoch(
                stimulus_start_time=self.session_start + datetime.timedelta(seconds=epoch_start),
                stimulus_end_time=self.session_start + datetime.timedelta(seconds=epoch_end),
                stimulus_name=epoch_name,
                software=[software_obj],
                script=script_obj,
                stimulus_modalities=[stim.VISUAL],
                stimulus_parameters=[params_obj],
            )
            schema_epochs.append(epoch_obj)

        return schema_epochs



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a session.json file for an ephys session')
    parser.add_argument('session_id', help='session ID (lims or np-exp foldername) or path to session folder')
    parser.add_argument('json-settings', help='json containing at minimum the fields "session_type" and "iacuc protocol"')
    return parser.parse_args()


def main() -> None:
    sessionETL = CamstimSession(**vars(parse_args()))
    sessionETL.generate_session_json()


if __name__ == '__main__':
    main()