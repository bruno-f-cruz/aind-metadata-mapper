"""Sets up the MRI ingest ETL"""

import logging
import traceback
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from aind_data_schema.base import AindCoreModel
from aind_data_schema.components.coordinates import (
    Rotation3dTransform,
    Scale3dTransform,
    Translation3dTransform,
)
from aind_data_schema.components.devices import (
    MagneticStrength,
    Scanner,
    ScannerLocation,
)
from aind_data_schema.core.session import (
    MRIScan,
    MriScanSequence,
    ScanType,
    Session,
    Stream,
    SubjectPosition,
)
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.units import TimeUnit
from bruker2nifti._metadata import BrukerMetadata
from pydantic import Field
from pydantic_settings import BaseSettings

from aind_metadata_mapper.core import GenericEtl, JobResponse


class JobSettings(BaseSettings):
    """Data that needs to be input by user."""

    data_path: Path
    output_directory: Optional[Path] = Field(
        default=None,
        description=(
            "Directory where to save the json file to. If None, then json"
            " contents will be returned in the Response message."
        ),
    )
    experimenter_full_name: List[str]
    session_type: str
    primary_scan_number: int
    setup_scan_number: int
    scanner_name: str
    scan_location: ScannerLocation
    magnetic_strength: MagneticStrength
    subject_id: str
    iacuc_protocol: str
    session_notes: str


PROTOCOL_ID = "placeholder mri protocol id"

DATETIME_FORMAT = "%H:%M:%S %d %b %Y"
LENGTH_FORMAT = "%Hh%Mm%Ss%fms"


class MRIEtl(GenericEtl[JobSettings]):
    """Class for MRI ETL process."""

    def __init__(self, job_settings: JobSettings):
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

    def _extract(self) -> BrukerMetadata:
        """Extract the data from the bruker files."""

        metadata = BrukerMetadata(self.job_settings.data_path)
        metadata.parse_scans()
        metadata.parse_subject()

        # self.n_scans = self.metadata.list_scans()
        return metadata

    def _transform(self, input_metadata) -> AindCoreModel:
        """Transform the data into the AIND data schema."""

        self.scan_data = input_metadata.scan_data
        self.subject_data = input_metadata.subject_data

        return self.load_mri_session(
            experimenter=self.job_settings.experimenter_full_name,
            primary_scan_number=self.job_settings.primary_scan_number,
            setup_scan_number=self.job_settings.setup_scan_number,
        )

    def run_job(self) -> JobResponse:
        """Run the job and return the response."""

        extracted = self._extract()
        transformed = self._transform(extracted)

        job_response = self._load(
            transformed, self.job_settings.output_directory
        )

        return job_response

    def load_mri_session(
        self,
        experimenter: str,
        primary_scan_number: str,
        setup_scan_number: str,
    ) -> Session:
        """Load the MRI session data into the AIND data schema."""

        scans = []
        for scan in self.scan_data.keys():
            scan_type = "3D Scan"
            if scan == setup_scan_number:
                scan_type = "Set Up"
            primary_scan = False
            if scan == primary_scan_number:
                primary_scan = True
            new_scan = self.make_model_from_scan(scan, scan_type, primary_scan)
            logging.info(f"loaded scan {new_scan}")

            scans.append(new_scan)

        logging.info(f"loaded scans: {scans}")

        start_time = datetime.strptime(self.scan_data[list(self.scan_data.keys())[0]]["acqp"]["ACQ_time"], DATETIME_FORMAT)
        final_scan_start = datetime.strptime(self.scan_data[list(self.scan_data.keys())[-1]]["acqp"]["ACQ_time"], DATETIME_FORMAT)
        final_scan_duration = datetime.strptime(self.scan_data[list(self.scan_data.keys())[-1]]["method"]["ScanTimeStr"], LENGTH_FORMAT)
        end_time = final_scan_start + timedelta(hours=final_scan_duration.hour, minutes=final_scan_duration.minute, seconds=final_scan_duration.second, microseconds=final_scan_duration.microsecond)

        stream = Stream(
            stream_start_time=datetime.now(),
            # This is probably the same as session start/end
            stream_end_time=datetime.now(),
            mri_scans=scans,
            stream_modalities=[Modality.MRI],
        )

        return Session(
            subject_id=self.job_settings.subject_id,
            session_start_time=start_time,  # see where to find this
            session_end_time=end_time,
            session_type=self.job_settings.session_type,
            experimenter_full_name=experimenter,
            protocol_id=[PROTOCOL_ID],
            iacuc_protocol=self.job_settings.iacuc_protocol,
            data_streams=[stream],
            rig_id=self.job_settings.scanner_name,
            mouse_platform_name="NA",
            active_mouse_platform=False,
            notes=self.job_settings.session_notes,
        )

    def make_model_from_scan(  # noqa: C901
        self, scan_index: str, scan_type, primary_scan: bool
    ) -> MRIScan:
        """load scan data into the AIND data schema."""

        logging.info(f"loading scan {scan_index}")

        self.cur_visu_pars = self.scan_data[scan_index]["recons"]["1"][
            "visu_pars"
        ]
        self.cur_method = self.scan_data[scan_index]["method"]

        subj_pos = self.subject_data["SUBJECT_position"]
        if "supine" in subj_pos.lower():
            subj_pos = "Supine"
        elif "prone" in subj_pos.lower():
            subj_pos = "Prone"

        scan_sequence = MriScanSequence.OTHER
        notes = None
        if "RARE" in self.cur_method["Method"]:
            scan_sequence = MriScanSequence(self.cur_method["Method"])
        else:
            notes = f"Scan sequence {self.cur_method['Method']} not recognized"

        rare_factor = None
        if "RareFactor" in self.cur_method.keys():
            rare_factor = self.cur_method["RareFactor"]

        if "EffectiveTE" in self.cur_method.keys():
            eff_echo_time = Decimal(self.cur_method["EffectiveTE"])
        else:
            eff_echo_time = None

        rotation = self.cur_visu_pars["VisuCoreOrientation"]
        if rotation.shape == (1, 9):
            rotation = Rotation3dTransform(rotation=rotation.tolist()[0])
        else:
            rotation = None

        translation = self.cur_visu_pars["VisuCorePosition"]

        if translation.shape == (1, 3):
            translation = Translation3dTransform(
                translation=translation.tolist()[0]
            )
        else:
            translation = None

        scale = self.cur_method["SpatResol"]
        if not isinstance(scale, list):
            scale = scale.tolist()

        if len(scale) == 3:
            scale = Scale3dTransform(scale=scale)
        else:
            scale = None

        try:
            return MRIScan(
                scan_index=scan_index,
                scan_type=ScanType(scan_type),  # set by scientists
                primary_scan=primary_scan,  # set by scientists
                mri_scanner=Scanner(
                    name=self.job_settings.scanner_name,
                    scanner_location=self.job_settings.scan_location,
                    magnetic_strength=self.job_settings.magnetic_strength,
                    magnetic_strength_unit="T",
                ),
                scan_sequence_type=scan_sequence,  # method ##$Method=RARE,
                rare_factor=rare_factor,  # method ##$PVM_RareFactor=8,
                echo_time=self.cur_method[
                    "EchoTime"
                ],  # method ##$PVM_EchoTime=0.01,
                effective_echo_time=eff_echo_time,  # method ##$EffectiveTE=(1)
                echo_time_unit=TimeUnit.MS,  # what do we want here?
                repetition_time=self.cur_method[
                    "RepetitionTime"
                ],  # method ##$PVM_RepetitionTime=500,
                repetition_time_unit=TimeUnit.MS,  # ditto
                vc_orientation=rotation,
                vc_position=translation,
                subject_position=SubjectPosition(subj_pos),
                voxel_sizes=scale,
                processing_steps=[],
                additional_scan_parameters={},
                notes=notes,
            )
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Error loading scan {scan_index}: {e}")
