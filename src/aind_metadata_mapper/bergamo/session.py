"""Module to map bergamo metadata into a session model."""

import argparse
import bisect
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from aind_data_schema.core.session import (
    DetectorConfig,
    FieldOfView,
    LaserConfig,
    Modality,
    Session,
    Stream, TriggerType,
)
from aind_data_schema.models.stimulus import (
    PhotoStimulation,
    PhotoStimulationGroup,
    StimulusEpoch,
)
from aind_data_schema.models.units import PowerUnit, SizeUnit, TimeUnit
from pydantic import Field
from pydantic_settings import BaseSettings
from ScanImageTiffReader import ScanImageTiffReader

from aind_metadata_mapper.core import GenericEtl, JobResponse


class JobSettings(BaseSettings):
    """Data that needs to be input by user. Can be pulled from env vars with
    BERGAMO prefix or set explicitly."""

    input_source: Path = Field(
        ..., description="Directory of files that need to be parsed."
    )
    output_directory: Optional[Path] = Field(
        default=None,
        description=(
            "Directory where to save the json file to. If None, then json"
            " contents will be returned in the Response message."
        ),
    )

    experimenter_full_name: List[str]
    subject_id: str

    # TODO: Look into whether defaults can be set for these fields
    mouse_platform_name: str
    active_mouse_platform: bool

    # Data that might change but can have default values
    session_type: str = "BCI"
    iacuc_protocol: str = "2115"
    rig_id: str = "Bergamo photostim."
    camera_names: List[str] = ["Side Camera"]
    laser_a_name: str = "Laser A"
    laser_a_wavelength: int = 920
    laser_a_wavelength_unit: SizeUnit = SizeUnit.NM
    detector_a_name: str = "PMT A"
    detector_a_exposure_time: Decimal = Decimal('0.1')
    detector_a_trigger_type: TriggerType = TriggerType.INTERNAL
    stimulus_name: str = "PhotoStimulation"
    fov_0_index: int = 0
    fov_0_imaging_depth: int = 150
    fov_0_targeted_structure: str = "M1"
    fov_0_coordinate_ml: Decimal = Decimal('1.5')
    fov_0_coordinate_ap: float = Decimal('1.5')
    fov_0_reference: str = "Bregma"
    fov_0_magnification: str = "16x"
    stream_modalities: List[Modality.ONE_OF] = [Modality.POPHYS]

    class Config:
        """Config to set env var prefix to BERGAMO"""

        env_prefix = "BERGAMO_"


class TifFileGroup(str, Enum):
    BEHAVIOR = "behavior"
    PHOTOSTIM = "photostim"
    SPONTANEOUS = "spontaneous"
    STACK = "stack"


@dataclass(frozen=True)
class RawImageInfo:
    """Raw metadata from a tif file"""

    reader_metadata_header: dict
    reader_metadata_json: dict
    # The reader descriptions for the last tif file
    reader_descriptions: List[dict]
    # Looks like [620, 800, 800]
    # [num_of_frames, pixel_width, pixel_height]?
    reader_shape: List[int]


@dataclass(frozen=True)
class ParsedMetadataInfo:
    """Tif file metadata that's needed downstream"""

    tif_file_group: TifFileGroup
    number_of_tif_files: int  # This should correspond to the number of trials
    h_photostim: dict
    h_roi_manager: dict
    h_beams: dict
    h_fast_z: dict
    imaging_roi_group: dict
    photostim_roi_groups: List[dict]
    reader_description_last: dict
    reader_shape: List[int]


class BergamoEtl(GenericEtl[JobSettings]):
    """Class to manage transforming bergamo data files into a Session object"""

    def __init__(
        self,
        job_settings: Union[JobSettings, str],
    ):
        """
        Class constructor for Base etl class.
        Parameters
        ----------
        job_settings: Union[JobSettings, str]
          Variables for a particular session
        """
        if isinstance(job_settings, str):
            job_settings_model = JobSettings.model_validate_json(job_settings)
        else:
            job_settings_model = job_settings
        super().__init__(job_settings=job_settings_model)

    def _get_tif_file_locations(self) -> Dict[str, List[Path]]:
        """Scans the input source directory and returns a dictionary of file
        groups in an ordered list. For example, if the directory had
        [neuron2_00001.tif, neuron2_00002.tif, stackPost_00001.tif,
        stackPost_00002.tif, stackPost_00003.tif], then it will return
        { "neuron2": [neuron2_00001.tif, neuron2_00002.tif],
         "stackPost":
           [stackPost_00001.tif, stackPost_00002.tif, stackPost_00003.tif]
        }
        """
        compiled_regex = re.compile(r"^(.*)_.*?(\d+).tif+$")
        tif_file_map = {}
        for root, dirs, files in os.walk(self.job_settings.input_source):
            for name in files:
                matched = re.match(compiled_regex, name)
                if matched:
                    groups = matched.groups()
                    file_stem = groups[0]
                    # tif_number = groups[1]
                    tif_filepath = Path(os.path.join(root, name))
                    if tif_file_map.get(file_stem) is None:
                        tif_file_map[file_stem] = [tif_filepath]
                    else:
                        bisect.insort(tif_file_map[file_stem], tif_filepath)

            # Only scan the top level files
            break
        return tif_file_map

    def _extract_raw_info_from_file(self, file_path: Path) -> RawImageInfo:
        with ScanImageTiffReader(str(file_path)) as reader:
            reader_metadata = reader.metadata()
            reader_shape = reader.shape()
            reader_descriptions = [
                dict(
                    [
                        (s.split(" = ", 1)[0], s.split(" = ", 1)[1])
                        for s in reader.description(i).strip().split("\n")
                    ]
                )
                for i in range(0, len(reader))
            ]

        metadata_first_part = reader_metadata.split("\n\n")[0]
        flat_metadata_header_dict = dict(
            [
                (s.split(" = ", 1)[0], s.split(" = ", 1)[1])
                for s in metadata_first_part.split("\n")
            ]
        )
        metadata_dict = self._flat_dict_to_nested(flat_metadata_header_dict)
        reader_metadata_json = json.loads(reader_metadata.split("\n\n")[1])
        # Move SI dictionary up one level
        if "SI" in metadata_dict.keys():
            si_contents = metadata_dict.pop("SI")
            metadata_dict.update(si_contents)
        return RawImageInfo(
            reader_shape=reader_shape,
            reader_metadata_header=metadata_dict,
            reader_metadata_json=reader_metadata_json,
            reader_descriptions=reader_descriptions,
        )

    @staticmethod
    def _map_raw_image_info_to_tif_file_group(
        raw_image_info: RawImageInfo,
    ) -> TifFileGroup:
        header = raw_image_info.reader_metadata_header
        if header.get("hPhotostim", {}).get("status") in [
            "'Running'",
            "Running",
        ]:
            return TifFileGroup.PHOTOSTIM
        elif (
            header.get("hIntegrationRoiManager", {}).get("enable") == "true"
            and header.get("hIntegrationRoiManager", {}).get(
                "outputChannelsEnabled"
            )
            == "true"
            and header.get("extTrigEnable", {}) == "1"
        ):
            return TifFileGroup.BEHAVIOR
        elif header.get("hStackManager", {}).get("enable") == "true":
            return TifFileGroup.STACK
        else:
            return TifFileGroup.SPONTANEOUS

    def _parse_raw_metadata(
        self, raw_image_info: RawImageInfo, number_of_files: int
    ) -> ParsedMetadataInfo:
        h_roi_manager = raw_image_info.reader_metadata_header.get(
            "hRoiManager", {}
        )
        h_beams = raw_image_info.reader_metadata_header.get("hBeams", {})
        h_fast_z = raw_image_info.reader_metadata_header.get("hFastZ", {})
        h_photostim = raw_image_info.reader_metadata_header.get(
            "hPhotostim", {}
        )
        roi_groups = raw_image_info.reader_metadata_json.get("RoiGroups", {})
        imaging_roi_group = roi_groups.get("imagingRoiGroup", {})
        photostim_roi_groups = roi_groups.get("photostimRoiGroups", [])

        reader_description_last = raw_image_info.reader_descriptions[-1]

        tif_file_group = self._map_raw_image_info_to_tif_file_group(
            raw_image_info=raw_image_info
        )

        return ParsedMetadataInfo(
            tif_file_group=tif_file_group,
            number_of_tif_files=number_of_files,
            h_photostim=h_photostim,
            h_roi_manager=h_roi_manager,
            h_beams=h_beams,
            h_fast_z=h_fast_z,
            imaging_roi_group=imaging_roi_group,
            photostim_roi_groups=photostim_roi_groups,
            reader_description_last=reader_description_last,
            reader_shape=raw_image_info.reader_shape,
        )

    def _extract_parsed_metadata_info_from_files(
        self, tif_file_locations: Dict[str, List[Path]]
    ) -> Dict[Tuple[str, TifFileGroup], ParsedMetadataInfo]:
        parsed_map = {}
        for file_stem, files in tif_file_locations.items():
            number_of_files = len(files)
            last_file = files[-1]
            raw_info = self._extract_raw_info_from_file(last_file)
            parsed_info = self._parse_raw_metadata(
                raw_image_info=raw_info, number_of_files=number_of_files
            )
            parsed_map[(file_stem, parsed_info.tif_file_group)] = parsed_info
        return parsed_map

    @staticmethod
    def _map_to_parsed_info_group_to_photo_stim_group(
        parsed_info_group: dict, list_index: int, number_of_trials: int
    ) -> PhotoStimulationGroup:
        number_of_neurons = int(
            np.array(
                parsed_info_group["rois"][1]["scanfields"]["slmPattern"]
            ).shape[0]
        )
        stimulation_laser_power = Decimal(
            str(parsed_info_group["rois"][1]["scanfields"]["powers"])
        )
        number_spirals = int(
            parsed_info_group["rois"][1]["scanfields"]["repetitions"]
        )
        spiral_duration = Decimal(
            str(parsed_info_group["rois"][1]["scanfields"]["duration"])
        )
        inter_spiral_interval = Decimal(
            str(parsed_info_group["rois"][2]["scanfields"]["duration"])
        )

        return PhotoStimulationGroup(
            group_index=list_index,
            number_of_neurons=number_of_neurons,
            stimulation_laser_power=stimulation_laser_power,
            stimulation_laser_power_unit=PowerUnit.PERCENT,
            number_trials=number_of_trials,
            number_spirals=number_spirals,
            spiral_duration=spiral_duration,
            inter_spiral_interval=inter_spiral_interval,
        )

    def _map_photo_stim_info_to_stimulus_epoch(
        self, photo_stim_info: ParsedMetadataInfo
    ) -> StimulusEpoch:

        # Number of trials should equal the number of tif files in the
        # photo_stim group?
        number_of_trials = photo_stim_info.number_of_tif_files
        sequence_stimulus = json.loads(
            photo_stim_info.h_photostim.get(
                "sequenceSelectedStimuli", "[]"
            ).replace(" ", ",")
        )
        number_of_groups = max(sequence_stimulus)
        # In theory, the number of groups should
        # match len(photostim_info.photostim_roi_groups)
        mapped_photostimulation_groups = [
            self._map_to_parsed_info_group_to_photo_stim_group(
                parsed_info_group=e[1],
                list_index=e[0],
                number_of_trials=number_of_trials,
            )
            for e in enumerate(photo_stim_info.photostim_roi_groups)
        ]
        # Look into this?
        inter_trial_interval = 1 / Decimal(
            photo_stim_info.h_roi_manager["scanFrameRate"]
        ) * photo_stim_info.reader_shape[0]
        stimulus_start_time = datetime.strptime(
            photo_stim_info.reader_description_last["epoch"],
            "[%Y %m %d %H %M %S.%f]",
        )
        elapsed_time = float(
            photo_stim_info.reader_description_last["frameTimestamps_sec"]
        )
        stimulus_end_time = stimulus_start_time + timedelta(
            seconds=elapsed_time
        )
        photo_stimulation = PhotoStimulation(
            stimulus_name=self.job_settings.stimulus_name,
            number_groups=number_of_groups,
            groups=mapped_photostimulation_groups,
            inter_trial_interval=inter_trial_interval,
        )

        return StimulusEpoch(
            stimulus_start_time=stimulus_start_time,
            stimulus_end_time=stimulus_end_time,
            stimulus=photo_stimulation,
        )

    def _map_photo_stim_info_to_streams(self, photo_stim_info: ParsedMetadataInfo) -> Stream:
        stream_start_time = datetime.strptime(
            photo_stim_info.reader_description_last["epoch"],
            "[%Y %m %d %H %M %S.%f]",
        )
        elapsed_time = float(
            photo_stim_info.reader_description_last["frameTimestamps_sec"]
        )
        stream_end_time = stream_start_time + timedelta(
            seconds=elapsed_time
        )
        laser_config = LaserConfig(
            name=self.job_settings.laser_a_name,  # Must match rig json
            wavelength =self.job_settings.laser_a_wavelength,
            excitation_power=Decimal(
                photo_stim_info.h_beams['powers'][1:-1].split()[0]
            ),
            excitation_power_unit = PowerUnit.PERCENT,
        )
        detector_config = DetectorConfig(
            name=self.job_settings.detector_a_name,
            exposure_time=self.job_settings.detector_a_exposure_time,
            exposure_time_unit=TimeUnit.S,
            trigger_type=self.job_settings.detector_a_trigger_type
        )
        ophys_fov = FieldOfView(
            index=0,
            imaging_depth=self.job_settings.fov_0_imaging_depth,
            targeted_structure=self.job_settings.fov_0_targeted_structure,
            fov_coordinate_ml=self.job_settings.fov_0_coordinate_ml,
            fov_coordinate_ap=self.job_settings.fov_0_coordinate_ap,
            fov_reference=self.job_settings.fov_0_reference,
            fov_width=int(
                photo_stim_info.h_roi_manager['pixelsPerLine']),
            fov_height=int(
                photo_stim_info.h_roi_manager['linesPerFrame']),
            magnification=self.job_settings.fov_0_magnification,
            fov_scale_factor=Decimal(
                photo_stim_info.h_roi_manager['scanZoomFactor']
            ),
            frame_rate=Decimal(
                photo_stim_info.h_roi_manager['scanFrameRate']),
        )
        camera_names = self.job_settings.camera_names
        return Stream(
            stream_start_time=stream_start_time,
            stream_end_time=stream_end_time,
            camera_names=camera_names,
            light_sources=[
                laser_config
            ],
            detectors=[detector_config],
            ophys_fovs=[ophys_fov],
            mouse_platform_name=self.job_settings.mouse_platform_name,
            active_mouse_platform=self.job_settings.active_mouse_platform,
            stream_modalities=self.job_settings.stream_modalities,
        )

    @staticmethod
    def _flat_dict_to_nested(flat: dict, key_delim: str = ".") -> dict:
        """
        Utility method to convert a flat dictionary into a nested dictionary.
        Modified from https://stackoverflow.com/a/50607551
        Parameters
        ----------
        flat : dict
          Example {"a.b.c": 1, "a.b.d": 2, "e.f": 3}
        key_delim : str
          Delimiter on dictionary keys. Default is '.'.

        Returns
        -------
        dict
          A nested dictionary like {"a": {"b": {"c":1, "d":2}, "e": {"f":3}}
        """

        def __nest_dict_rec(k, v, out) -> None:
            """Simple recursive method being called."""
            k, *rest = k.split(key_delim, 1)
            if rest:
                __nest_dict_rec(rest[0], v, out.setdefault(k, {}))
            else:
                out[k] = v

        result = {}
        for flat_key, flat_val in flat.items():
            __nest_dict_rec(flat_key, flat_val, result)
        return result

    def _transform(self, parsed_data: Dict[Tuple[str, TifFileGroup], ParsedMetadataInfo]) -> Session:

        photo_stim_file_info = [
            (k, v)
            for k, v in parsed_data.items()
            if k[1] == TifFileGroup.PHOTOSTIM
        ]
        # There should only be one photo_stim group? We can add an assertion
        photo_stim_info = photo_stim_file_info[0][1]
        stimulus_epoch = self._map_photo_stim_info_to_stimulus_epoch(photo_stim_info=photo_stim_info)
        stream = self._map_photo_stim_info_to_streams(photo_stim_info=photo_stim_info)

        return Session(
            experimenter_full_name=self.job_settings.experimenter_full_name,
            session_start_time=stream.stream_start_time,
            session_end_time=stream.stream_end_time,
            session_type=self.job_settings.session_type,
            iacuc_protocol=self.job_settings.iacuc_protocol,
            rig_id=self.job_settings.rig_id,
            subject_id=self.job_settings.subject_id,
            animal_weight_prior=None,
            animal_weight_post=None,
            data_streams=[stream],
            stimulus_epochs=[stimulus_epoch],
        )

    def run_job(self) -> JobResponse:
        """Run the etl job and return a JobResponse."""
        tif_locations = self._get_tif_file_locations()
        parsed_data = self. _extract_parsed_metadata_info_from_files(
            tif_file_locations=tif_locations
        )
        session = self._transform(parsed_data=parsed_data)
        job_response = self._load(
            session, self.job_settings.output_directory
        )
        return job_response

    # TODO: The following can probably be abstracted
    @classmethod
    def from_args(cls, args: list):
        """
        Adds ability to construct settings from a list of arguments.
        Parameters
        ----------
        args : list
        A list of command line arguments to parse.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-j",
            "--job-settings",
            required=True,
            type=str,
            help=(
                r"""
                Custom settings defined by the user defined as a json
                 string. For example: -j
                 '{
                 "input_source":"/directory/to/read/from",
                 "output_directory":"/directory/to/write/to",
                 "experimenter_full_name":["John Smith","Jane Smith"],
                 "subject_id":"12345",
                 "session_start_time":"2023-10-10T10:10:10",
                 "session_end_time":"2023-10-10T18:10:10",
                 "stream_start_time": "2023-10-10T11:10:10",
                 "stream_end_time":"2023-10-10T17:10:10",
                 "stimulus_start_time":"12:10:10",
                 "stimulus_end_time":"13:10:10"}'
                """
            ),
        )
        job_args = parser.parse_args(args)
        job_settings_from_args = JobSettings.model_validate_json(
            job_args.job_settings
        )
        return cls(
            job_settings=job_settings_from_args,
        )


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    etl = BergamoEtl.from_args(sys_args)
    etl.run_job()
