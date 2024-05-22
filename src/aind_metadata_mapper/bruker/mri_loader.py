"""Sets up the MRI ingest ETL"""

from bruker2nifti._metadata import BrukerMetadata

from pathlib import Path
from aind_data_schema.components.coordinates import Rotation3dTransform, Scale3dTransform, Translation3dTransform
from aind_data_schema.core.session import MRIScan, Session, MriScanSequence, ScanType, SubjectPosition, Stream
from decimal import Decimal
from aind_data_schema_models.units import MassUnit, TimeUnit
from aind_data_schema.components.devices import Scanner, ScannerLocation, MagneticStrength
from datetime import datetime
from pydantic_settings import BaseSettings
from aind_metadata_mapper.core import GenericEtl, JobResponse
from dataclasses import dataclass
from aind_data_schema.base import AindCoreModel
from typing import Any, Generic, Optional, TypeVar, Union
from aind_data_schema_models.modalities import Modality

import traceback
import logging

from pydantic import Field
from typing import List, Optional, Union


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
    primary_scan_number: int
    setup_scan_number: int
    scanner_name: str
    scan_location: ScannerLocation
    magnetic_strength: MagneticStrength
    subject_id: str
    protocol_id: str
    iacuc_protocol: str
    notes: str


# @dataclass(frozen=True)
# class ExtractedMetadata:
#     """Raw Bruker data gets parsted here."""

#     metadata: BrukerMetadata
#     n_scans: List[str]


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
            scan_location=self.job_settings.scan_location,
            magnet_strength=self.job_settings.MagneticStrength
        )

    def run_job(self) -> JobResponse:
        """Run the job and return the response."""

        extracted = self._extract()
        transformed = self._transform(extracted)

        job_response = self._load(
            transformed,
            self.job_settings.output_directory
        )

        return job_response

    def load_mri_session(self, experimenter: str, primary_scan_number: str, setup_scan_number: str) -> Session:
        """Load the MRI session data into the AIND data schema."""

        scans = []
        for scan in self.scan_data.keys():
            print("SCAN: ", scan)
            scan_type = "3D Scan"
            if scan == setup_scan_number:
                scan_type = "Set Up"
            primary_scan = False
            if scan == primary_scan_number:
                primary_scan = True
            new_scan = self.make_model_from_scan(scan, scan_type, primary_scan)
            logging.info(f'loaded scan {new_scan}')

            print("dumped: ", new_scan.model_dump_json())
            scans.append(new_scan)


        logging.info(f'loaded scans: {scans}')

        stream = Stream(
            stream_start_time=,
            stream_end_time=,
            mri_scans=scans,
            stream_modalities=[Modality.MRI]
        )

        return Session(
            subject_id=self.job_settings.subject_id,
            session_start_time=,
            session_end_time=,
            experimenter_full_name=experimenter, 
            protocol_id=[self.job_settings.protocol_id],
            iacuc_protocol=self.job_settings.iacuc_protocol,
            data_streams=[stream],
            mouse_platform_name="NA",
            notes="none"
        )
    

    def make_model_from_scan(self, scan_index: str, scan_type, primary_scan: bool) -> MRIScan:
        """load scan data into the AIND data schema."""
        
        logging.info(f'loading scan {scan_index}')   

        self.cur_visu_pars = self.scan_data[scan_index]['recons']['1']['visu_pars']
        self.cur_method = self.scan_data[scan_index]['method']

        subj_pos = self.subject_data["SUBJECT_position"]
        if 'supine' in subj_pos.lower():
            subj_pos = 'Supine'
        elif 'prone' in subj_pos.lower():
            subj_pos = 'Prone'

        scan_sequence = MriScanSequence.OTHER
        notes = None
        if 'RARE' in self.cur_method['Method']:
            scan_sequence = MriScanSequence(self.cur_method['Method'])
        else:
            notes = f"Scan sequence {self.cur_method['Method']} not recognized"

        rare_factor = None
        if 'RareFactor' in self.cur_method.keys():
            rare_factor = self.cur_method['RareFactor']

        if 'EffectiveTE' in self.cur_method.keys():
            eff_echo_time = Decimal(self.cur_method['EffectiveTE'])
        else:
            eff_echo_time = None

        rotation=self.cur_visu_pars['VisuCoreOrientation']
        if rotation.shape == (1,9):
            rotation=Rotation3dTransform(rotation=rotation.tolist()[0])
        else:
            rotation = None
        
        translation=self.cur_visu_pars['VisuCorePosition']

        if translation.shape == (1,3):
            translation=Translation3dTransform(translation=translation.tolist()[0])
        else:
            translation = None

        print("spatreso: ", self.cur_method['SpatResol'])
        scale=self.cur_method['SpatResol']
        if not isinstance(scale, list):
            scale = scale.tolist()
        while len(scale) < 3: # TODO: THIS IS NOT THE IDEAL SOLUTION, talk to scientists about what to do for too few items in spatreso list
            scale.append(0)
        
        scale = Scale3dTransform(scale=scale)

        try:
            return MRIScan(
                scan_index=scan_index,
                scan_type=ScanType(scan_type), # set by scientists
                primary_scan=primary_scan, # set by scientists
                mri_scanner=Scanner(
                    name=self.job_settings.scanner_name,
                    scanner_location=self.job_settings.scan_location,
                    magnetic_strength=self.job_settings.magnet_strength, 
                    magnetic_strength_unit="T", 
                ),
                scan_sequence_type=scan_sequence, # method ##$Method=RARE,
                rare_factor=rare_factor, # method ##$PVM_RareFactor=8,
                echo_time=self.cur_method['EchoTime'], # method ##$PVM_EchoTime=0.01,
                effective_echo_time=eff_echo_time, # method ##$EffectiveTE=(1)
                # echo_time_unit=TimeUnit(), # what do we want here?
                repetition_time=self.cur_method['RepetitionTime'], # method ##$PVM_RepetitionTime=500,
                # repetition_time_unit=TimeUnit(), # ditto
                vc_orientation=rotation,# visu_pars  ##$VisuCoreOrientation=( 1, 9 )
                vc_position=translation, # visu_pars ##$VisuCorePosition=( 1, 3 )
                subject_position=SubjectPosition(subj_pos), # subject ##$SUBJECT_position=SUBJ_POS_Supine,
                voxel_sizes=scale, # method ##$PVM_SpatResol=( 3 )
                processing_steps=[],
                additional_scan_parameters={},
                notes=notes, # Where should we pull these?
            )      
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f'Error loading scan {scan_index}: {e}') 