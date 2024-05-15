"""
File containing Camstim class
"""

import argparse
import datetime
import io
import json
from pathlib import Path

import aind_data_schema
import aind_data_schema.core.session as session_schema
import np_session
import npc_sync
import numpy as np
import pandas as pd
from utils import pickle_functions as pkl_utils


class Camstim:
    """
    Methods used to extract stimulus epochs
    """

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

        self.pkl_path = self.npexp_path / f"{self.folder}.stim.pkl"
        self.opto_table_path = (
            self.npexp_path / f"{self.folder}_opto_epochs.csv"
        )
        self.stim_table_path = (
            self.npexp_path / f"{self.folder}_stim_epochs.csv"
        )
        self.sync_path = self.npexp_path / f"{self.folder}.sync"

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


    def epoch_from_opto_table(self) -> session_schema.StimulusEpoch:
        """
        From the optogenetic stimulation table, returns a single schema
        stimulus epoch representing the optotagging period. Include all
        unknown table columns (not start_time, stop_time, stim_name) as
        parameters, and include the set of all of that column's values as the
        parameter values.
        """
        stim = aind_data_schema.core.session.StimulusModality

        script_obj = aind_data_schema.models.devices.Software(
            name=self.mtrain["regimen"]["name"],
            version="1.0",
            url=self.mtrain["regimen"]["script"],
        )

        opto_table = pd.read_csv(self.opto_table_path)

        opto_params = {}
        for column in opto_table:
            if column in ("start_time", "stop_time", "stim_name"):
                continue
            param_set = set(opto_table[column].dropna())
            opto_params[column] = param_set

        params_obj = session_schema.VisualStimulation(
            stimulus_name="Optogenetic Stimulation",
            stimulus_parameters=opto_params,
            stimulus_template_name=[],
        )

        opto_epoch = session_schema.StimulusEpoch(
            stimulus_start_time=self.session_start
            + datetime.timedelta(seconds=opto_table.start_time.iloc[0]),
            stimulus_end_time=self.session_start
            + datetime.timedelta(seconds=opto_table.start_time.iloc[-1]),
            stimulus_name="Optogenetic Stimulation",
            software=[],
            script=script_obj,
            stimulus_modalities=[stim.OPTOGENETICS],
            stimulus_parameters=[params_obj],
        )

        return opto_epoch

    def extract_stim_epochs(
        self, stim_table: pd.DataFrame
    ) -> list[list[str, int, int, dict, set]]:
        """
        Returns a list of stimulus epochs, where an epoch takes the form
        (name, start, stop, params_dict, template names). Iterates over the
        stimulus epochs table, identifying epochs based on when the
        'stim_name' field of the table changes.

        For each epoch, every unknown column (not start_time, stop_time,
        stim_name, stim_type, or frame) are listed as parameters, and the set
        of values for that column are listed as parameter values.
        """
        epochs = []

        current_epoch = [None, 0.0, 0.0, {}, set()]
        epoch_start_idx = 0
        for current_idx, row in stim_table.iterrows():
            # if the stim name changes, summarize current epoch's parameters
            # and start a new epoch
            if row["stim_name"] != current_epoch[0]:
                for column in stim_table:
                    if column not in (
                        "start_time",
                        "stop_time",
                        "stim_name",
                        "stim_type",
                        "frame",
                    ):
                        param_set = set(
                            stim_table[column][
                                epoch_start_idx:current_idx
                            ].dropna()
                        )
                        current_epoch[3][column] = param_set

                epochs.append(current_epoch)
                epoch_start_idx = current_idx
                current_epoch = [
                    row["stim_name"],
                    row["start_time"],
                    row["stop_time"],
                    {},
                    set(),
                ]
            # if stim name hasn't changed, we are in the same epoch, keep
            # pushing the stop time
            else:
                current_epoch[2] = row["stop_time"]

            # if this row is a movie or image set, record it's stim name in
            # the epoch's templates entry
            if (
                "image" in row.get("stim_type", "").lower()
                or "movie" in row.get("stim_type", "").lower()
            ):
                current_epoch[4].add(row["stim_name"])

        # slice off dummy epoch from beginning
        return epochs[1:]

    def epochs_from_stim_table(self) -> list[session_schema.StimulusEpoch]:
        """
        From the stimulus epochs table, return a list of schema stimulus
        epochs representing the various periods of stimulus from the session.
        Also include the camstim version from pickle file and stimulus script
        used from mtrain.
        """
        stim = aind_data_schema.core.session.StimulusModality

        software_obj = aind_data_schema.components.devices.Software(
            name="camstim",
            version=pkl_utils.load_pkl(self.pkl_path)["platform"][
                "camstim"
            ].split("+")[0],
            url="https://eng-gitlab.corp.alleninstitute.org/braintv/camstim",
        )

        script_obj = aind_data_schema.components.devices.Software(
            name=self.mtrain["regimen"]["name"],
            version="1.0",
            url=self.mtrain["regimen"]["script"],
        )

        schema_epochs = []
        for (
            epoch_name,
            epoch_start,
            epoch_end,
            stim_params,
            stim_template_names,
        ) in self.extract_stim_epochs(pd.read_csv(self.stim_table_path)):
            params_obj = session_schema.VisualStimulation(
                stimulus_name=epoch_name,
                stimulus_parameters=stim_params,
                stimulus_template_name=stim_template_names,
            )

            epoch_obj = session_schema.StimulusEpoch(
                stimulus_start_time=self.session_start
                + datetime.timedelta(seconds=epoch_start),
                stimulus_end_time=self.session_start
                + datetime.timedelta(seconds=epoch_end),
                stimulus_name=epoch_name,
                software=[software_obj],
                script=script_obj,
                stimulus_modalities=[stim.VISUAL],
                stimulus_parameters=[params_obj],
            )
            schema_epochs.append(epoch_obj)

        return schema_epochs