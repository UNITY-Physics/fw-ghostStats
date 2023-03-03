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
                'CuSO4 Solutions @ 3.0T':{'name':'CuSO4_3T', 'head':3, 'tail':3},
                'CMRI LC Values':{'name':'CMRI_LC', 'head':3, 'tail':0}}
    
    data = {}
    for sheet_key in sheets.keys():
        df = pd.read_excel(fname, sheet_name=sheet_key, header=sheets[sheet_key]['head'])
        data[sheets[sheet_key]['name']] = df[:-1*sheets[sheet_key]['tail']]

    return data


class Calibration():

    def __init__(self, fname):
        self.data = read_calibration_sheet(fname)
        self.__sheets = {'T1': 'NiCl_3T', 'T2': 'MnCl_3T', 'ADC': 'CuSO4_3T'}

    def __get_vals(self, temp, mimics, column):
        """Get T1 or T2 values for a given temperature, sorted from lowest to highest concentration

        Args:
            temp (int): Temperature in degrees Celsius
            mimics (str): T1 or T2.
            column (str): T1 or T2.

        Returns:
            list: List of T1 or T2 values for a given temperature, sorted from lowest to highest concentration
        """
        try:
            sheet = self.__sheets.get(mimics)
        except KeyError:
            print('Mimics must be either T1 or T2')
        else:
            df = self.data.get(sheet)
            if temp not in range(16, 27, 2):
                raise ValueError('Temperature not available. Available temperatures are 16, 18, 20, 22, 24 and 26 degrees Celsius')
            else:
                return df.loc[df['Temperature (C)'] == temp, column + ' (ms)'].values

    
    def get_T1_conc(self):
        """Get NiCl concentrations
        
        Returns:
            list: List of NiCl concentrations
        
        Example:
            >>> calib = Calibration('calibration.xls')
            >>> calib.get_T1_conc()
        """
        return self.data.get('NiCl_15T').get('Concentration (mM)').values

    def get_T2_conc(self):
        """Get MnCl concentrations
        
        Returns:
            list: List of MnCl concentrations
            
        Example:
            >>> calib = Calibration('calibration.xls')
            >>> calib.get_T2_conc()
        """
        return self.data.get('MnCl_15T').get('Concentration (mM)').values

    def get_T1_vals(self, temp, mimics='T1'):
        """Get T1 values for a given temperature, sorted from lowest to highest concentration

        Args:
            temp (int): Temperature in degrees Celsius
            mimics (str, optional): T1 or T2. Defaults to 'T1'.

        Returns:
            list: List of T1 values for a given temperature, sorted from lowest to highest concentration
        
        Example:
            >>> calib = Calibration('calibration.xls')
            >>> calib.get_T1_vals(20)
        """
        return self.__get_vals(temp, mimics, 'T1')

    def get_T2_vals(self, temp, mimics='T2'):
        """Get T2 values for a given temperature, sorted from lowest to highest concentration
        
        Args:
            temp (int): Temperature in degrees Celsius
            mimics (str, optional): T1 or T2. Defaults to 'T2'.
        
        Returns:
            list: List of T2 values for a given temperature, sorted from lowest to highest concentration
        
        Example:
            >>> calib = Calibration('calibration.xls')
            >>> calib.get_T2_vals(20)
        """
        return self.__get_vals(temp, mimics, 'T1')


    def get_ADC_vals(self, temp):
        pass

    def get_LC_vals(self):
        pass