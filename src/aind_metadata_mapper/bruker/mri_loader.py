from bruker2nifti._metadata import BrukerMetadata
from pathlib import Path
from aind_data_schema.models.coordinates import Rotation3dTransform, Scale3dTransform, Translation3dTransform
from aind_data_schema.core.mri_session import MRIScan, MriSession, MriScanSequence, ScanType, SubjectPosition
from decimal import Decimal
from aind_data_schema.models.units import MassUnit, TimeUnit
from aind_data_schema.models.devices import Scanner, ScannerLocation, MagneticStrength
from datetime import datetime
from pydantic_settings import BaseSettings


import traceback
import logging

from pydantic import Field
from typing import List, Optional, Union


class JobSettings(BaseSettings):
    """Data that needs to be input by user."""

    output_directory: Optional[Path] = Field(
        default=None,
        description=(
            "Directory where to save the json file to. If None, then json"
            " contents will be returned in the Response message."
        ),
    )

    string_to_parse: str
    experimenter_full_name: List[str]
    primary_scan_number: int
    setup_scan_number: int
    scan_location: ScannerLocation
    MagneticStrength: MagneticStrength
    notes: str



class MRIEtl(GenericEtl[JobSettings]):
    def __init__(self, data_path):
        self.metadata = BrukerMetadata(data_path)
        self.metadata.parse_scans()
        self.metadata.parse_subject()
        self.n_scans = self.metadata.list_scans()


    def load_mri_session(self, experimenter: str, primary_scan_number: str, setup_scan_number: str, scan_location: ScannerLocation, magnet_strength: MagneticStrength) -> MRIScan:

        scans = []
        for scan in self.n_scans:
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

        return MriSession(
            subject_id="",
            session_start_time=datetime.now(), 
            session_end_time=datetime.now(),
            experimenter_full_name=experimenter, 
            protocol_id="",
            iacuc_protocol="",
            mri_scanner=Scanner(
                name="test_scanner",
                scanner_location=scan_location,
                magnetic_strength=magnet_strength, 
                magnetic_strength_unit="T", 
            ),
            scans=scans,
            notes="none"
        )
    

    def make_model_from_scan(self, scan_index: str, scan_type, primary_scan: bool) -> MRIScan:
        logging.info(f'loading scan {scan_index}')   

        self.cur_visu_pars = self.metadata.scan_data[scan_index]['recons']['1']['visu_pars']
        self.cur_method = self.metadata.scan_data[scan_index]['method']

        subj_pos = self.metadata.subject_data["SUBJECT_position"]
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

        scale=self.cur_method['SpatResol'].tolist()
        # while len(scale) < 3:
        #     scale.append(0)
        
        scale = Scale3dTransform(scale=scale)

        try:
            return MRIScan(
                scan_index=scan_index,
                scan_type=ScanType(scan_type), # set by scientists
                primary_scan=primary_scan, # set by scientists
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