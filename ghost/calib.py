import pandas as pd

"""
Everything that has to do with the phantom calibration.
"""

def read_calibration_sheet(fname, sheets=None):
    """Parse Caliber MRI calibration sheet

    Args:
        fname (str): Full path to calibration .xls sheet
        sheets (str, optional): Sheet description. Defaults to None.

    Returns:
        dict: Dictionary of dataframes with calibration data
    """
    if not sheets:
        sheets = {'ADC Solutions @ 1.5T':{'name':'ADC_15T', 'head':3, 'tail':2},
                'ADC Solutions @ 3.0T':{'name':'ADC_3T', 'head':3, 'tail':2},
                'NiCl Solutions @ 1.5T':{'name':'NiCl_15T', 'head':3, 'tail':2},
                'NiCl Solutions @ 3.0T':{'name':'NiCl_3T', 'head':3, 'tail':2},
                'MnCl Solutions @ 1.5T':{'name':'MnCl_15T', 'head':3, 'tail':2},
                'MnCl Solutions @ 3.0T':{'name':'MnCl_3T', 'head':3, 'tail':2},
                'CuSO4 Solutions @ 3.0T':{'name':'CuS04_3T', 'head':3, 'tail':3},
                'CMRI LC Values':{'name':'CMRI_LC', 'head':3, 'tail':0}}
    
    data = {}
    for sheet_key in sheets.keys():
        df = pd.read_excel(fname, sheet_name=sheet_key, header=sheets[sheet_key]['head'])
        data[sheets[sheet_key]['name']] = df[:-1*sheets[sheet_key]['tail']]

    return data


class Calibration():

    def __init__(self, fname):
        self.data = read_calibration_sheet(fname)

    def get_T1_conc(self):
        pass

    def get_T2_conc(self):
        pass

    def get_T1_vals(self, B0, temp):
        pass

    def get_T2_vals(self, B0, temp):
        pass

    def get_ADC_vals(self, temp):
        pass

    def get_LC_vals(self):
        pass