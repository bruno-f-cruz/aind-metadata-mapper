from aind_metadata_mapper.bruker.mri_loader import MRILoader
from pathlib import Path
from glob import glob

paths = glob("C:\\Users\\mae.moninghoff\\Documents\GitHub\\aind-data-schema-sphinx\\aind-metadata-mapper\\src\\aind_metadata_mapper\\bruker\\MRI_ingest\\MRI_files\\RawData2023_06_29")


for path in paths:
    loader = MRILoader(path)
scan7 = loader.make_model_from_scan('5', '3D Scan', True)
print(scan7)

session = loader.load_mri_session(["Mae"], "7", "1", ScannerLocation.FRED_HUTCH, MagneticStrength.MRI_7T)


session.write_standard_file(output_directory=Path("./output"), prefix=Path("test"))