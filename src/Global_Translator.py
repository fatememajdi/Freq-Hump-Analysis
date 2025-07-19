import re
import os
import logging
import traceback
# from MachWise import Wise
from datetime import datetime
# from API_spare_part import get_spare_part
# from Global_Data_List import data_list

# ==============================================================================================================================================================================================
# ==============================================================================================================================================================================================
# Methodes =====================================================================================================================================================================================

# Get type of file
def get_type(file_name):
    """
    Determines the type of file based on its name.
    
    :param file_name: The name of the file to analyze.
    :return: The type of file (integer)
    """
    suffix = ' - Time Signal.txt'
    file_name = file_name[:-len(suffix)]
    if ',' in file_name:
        return 6
    # elif '_' in file_name:
    #     return 5
    elif '.' in file_name:
        return 4
    elif ',' in file_name:
        return 1
    elif '-' in file_name:
        return 4
    else:
        return 2

# Regex for file names
def regex(file_path : str, server_name = 'demo'):
    """
    Determines the type of file based on its name.
    
    :param file_path: The path to the file to analyze.
    :param server_name: The name of the server (default is 'demo').
    :return: The type of file (integer)
    """
    #==================================== regex file names type one ============================================================
    check_str = re.findall(r".*\/(.*)",file_path)
    typ = get_type(check_str)
    spare_part = list()
    #==================================== regex file names type One ============================================================
    if typ == 2:
        name = re.findall(r".*\/(.*)-\d\w+\s+\w+-\w+_.*",file_path)
        point_direction =re.findall(r".*\/.*\s(.*)\_\s",file_path)
        temp_point = point_direction[0].split('-')
        point_direction = [temp_point[1]]
        parameter = re.findall(r".*\/.*\s(...)\s",file_path)
        date = re.findall(r".*\/.*\((.*)\s\d",file_path)
        hour = re.findall(r".*\/.*\(.*\s(.*)\)\s\-",file_path)
        temp = date[0].split('_')
        date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
        temp = hour[0].split('_')
        hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
    #==================================== regex file names type Two ============================================================
    elif typ == 1:
        name = re.findall(r".*\/([^\s]*).*",file_path)
        point_direction =re.findall(r".*\/.*\s.+(\d.+)\_\s",file_path)
        parameter = re.findall(r".*\/.*\s(...)\s",file_path)
        date = re.findall(r".*\/.*\((.*)\s\d",file_path)
        hour = re.findall(r".*\/.*\(.*\s(.*)\)\s\-",file_path)
        temp = date[0].split('_')
        date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
        temp = hour[0].split('_')
        hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
    #==================================== regex file names type Three ============================================================
    elif typ == 3:
        name = re.findall(r".*\/(.*)\..*\s",file_path)
        point_direction =re.findall(r".*\/.*\s.*\-(\d.*)\_\s.*",file_path)
        if '-' in point_direction[0]:
            point_direction[0] = point_direction[0].replace('-', '')
        spare_part =re.findall(r".*\/.*\s(.*)\-\d.+\_\s",file_path)
        parameter = re.findall(r".*\/.*\s(...)\s",file_path)
        date = re.findall(r".*\/.*\((.*)\s\d",file_path)
        hour = re.findall(r".*\/.*\(.*\s(.*)\)\s\-",file_path)
        temp = date[0].split('_')
        date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
        temp = hour[0].split('_')
        hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
#==================================== regex file names type four ============================================================
    elif typ == 4:
        name = re.findall(r".*\/(.*)\-.*",file_path)
        if '-' in name[0]:
            name[0] = name[0].split('-')[0]
        point_direction =re.findall(r".*\/.*\s.*\s\w\-(.*)\_\s.*",file_path)
        if '-' in point_direction[0]:
            point_direction[0] = point_direction[0].split('-')[0]
        spare_part =re.findall(r".*\/.*\s(.*)\-\d.+\_\s",file_path)
        parameter = re.findall(r".*\/.*\s(...)\s",file_path)
        date = re.findall(r".*\/.*\((.*)\s\d",file_path)
        hour = re.findall(r".*\/.*\(.*\s(.*)\)\s\-",file_path)
        temp = date[0].split('_')
        date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
        temp = hour[0].split('_')
        hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
    #==================================== regex file names type four ============================================================
    elif typ == 5:
        name = re.findall(r".*\/(.*)\..*\s",file_path)
        point_direction = re.findall(r".*\/.*\..*.\s(.*)\-.*\_.*",file_path)
        spare_part = re.findall(r".*\/.*\..*.\s.*\-(.*)\_\s.*",file_path)
        parameter = re.findall(r".*\/.*\s(...)\s",file_path)
        date = re.findall(r".*\/.*\((.*)\s\d",file_path)
        hour = re.findall(r".*\/.*\(.*\s(.*)\)\s\-",file_path)
        temp = date[0].split('_')
        date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
        temp = hour[0].split('_')
        hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
        temp = point_direction
        point_direction = spare_part
        spare_part = temp
    elif typ == 6:
        name = re.findall(r".*\/(.*)\..*\s",file_path)
        # spare_dic = get_data(name, server_name)
        # set_spare = set()
        # for key in spare_dic[1].keys():
        #     temp = key.split(' ')
        #     set_spare.add(temp[0])
        point_direction = re.findall(r".*\/.*\..*.\s(.*)\_\s.*",file_path)
        parameter = re.findall(r".*\/.*\s(...)\s",file_path)
        date = re.findall(r".*\/.*\((.*)\s\d",file_path)
        hour = re.findall(r".*\/.*\(.*\s(.*)\)\s\-",file_path)
        temp = date[0].split('_')
        date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
        temp = hour[0].split('_')
        hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
        spare_part = list()
        if point_direction[0][0] == '3' or point_direction[0][0] == '4':
            spare_part.append('P')
        else:
            spare_part.append('M')
    else:
        print("Invalid Time Signal name.!")
    #==================================== Return data ==========================================================================
    # print("----------------------------------------------------------------")
    # print(check_str)
    name[0] = name[0].strip()
    # get_data(name[0])

    # Convert date to correct forme ==========================
    # Parse the string into a date object
    date_object = datetime.strptime(date[0], "%Y-%m-%d")
    # Format it to the desired format
    date[0] = date_object.strftime("%Y-%m-%d")
    # =========================================================
    # if name[0] == '2403':
    #     name[0] = '3111FN1'
    # print('========')
    # print('========')
    # print(name , point_direction, parameter, date, hour, spare_part)
    # print('========')
    # print('========')
    if '10' in point_direction[0] or '11' in point_direction[0] or '12' in point_direction[0] or '13' in point_direction[0] or '14' in point_direction[0]:
        point_direction[0] = [point_direction[0][0:2], point_direction[0][2]]

    # Return result
    return make_result_dictionary(name[0] , point_direction[0][0], point_direction[0][1], parameter[0], date[0], hour[0], spare_part[0])

# ==============================================================================================================================================================================================
# ==============================================================================================================================================================================================
# ==============================================================================================================================================================================================

#===============================================================================================================================
# Extra function ==============================================================================================================
def machine_name_list(signals_directory_path : str):
    """
    This function return the list of machines in directory to process.\n
    Input = {signals_directory_path : ADDRESS OF DIRECTORY INCLUDED SIGNALS.}\n
    Output = {LIST OF MACHINE NAMES IN DIRECTORY.}
    """
    equipments_list = list()
    signals = data_list(signals_directory_path)
    for signal in signals:
        translated = translator(signal)
        equipments_list.append(translated['name'])
    
    return list(set(equipments_list))

# Translator functions ==============================================================================================================
# Find just signal name from adress file
def find_signal_name(file_path):
    """
    input = address of txt file.\n
    output = return just name of files without directory of signal
    """
    name = os.path.basename(file_path)
    return name

# Normalize the date
def striptime(date):
    """
    input = date
    output = bst format for model
    """
     # Convert date to correct forme ==========================
    # Parse the string into a date object
    date_object = datetime.strptime(date, "%Y-%m-%d")
    # Format it to the desired format
    return date_object.strftime("%Y-%m-%d")
    # =========================================================
    
# Make result dictionart
def make_result_dictionary(name , point, direction, assignment, date, hour, spare_part):
    """
    Make a standard output.\n
    input = {name , point_direction, parameter, date, hour, spare_part}
    """
    return {'name' : name, 'point' : point, 'direction' : direction, 'spare_part' : spare_part, 'assignment' : assignment, 'date' : date, 'hour' : hour}

# Find spare from static informations
def find_spare(name, point):
    """
    Find spare from static informations
    """
    # Get machine informations
    info = Wise(name[0])
    spare_part_list = info.machine_part.keys()
    for spare in spare_part_list:
        if point in info.machine_part[spare]:
            return spare

# Normalize assignmet
def normalize_assignment(assignment : str):
    """
    Normalize assignment
    """
    return assignment.lower().capitalize()
    

#===============================================================================================================================
# Regex Per Company Section ====================================================================================================

def Zave_Cement(file_name):
    """
    Translate Zave Cement file names
    """
    #==================================== regex file names type one ============================================================
    signal_name = find_signal_name(file_name)
    # ==========================================================================================================================
    name = re.findall(r"(.*)-\d\w+\s+\w+-\w+_.*",signal_name)
    name[0] = name[0].strip()
    point_direction = re.findall(r".*\s.*\-(.*)\_\s",signal_name)
    # Seperate poit from direction =============================================================================================
    point = point_direction[0][0]
    direction = point_direction[0][1]
    # ==========================================================================================================================
    spare_part = re.findall(r".*\s(.*)\-.*\_\s",signal_name)
    assignment = re.findall(r".*\s(...)\s",signal_name)
    assignment = normalize_assignment(assignment[0])
    date = re.findall(r".*\((.*)\s\d",signal_name)
    hour = re.findall(r".*\(.*\s(.*)\)\s\-",signal_name)
    temp = date[0].split('_')
    date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
    temp = hour[0].split('_')
    hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
    name[0] = name[0].strip()
    date = striptime(date[0])
    #==================================== Return data ==========================================================================
    return make_result_dictionary(name[0] , point, direction, assignment, date, hour[0], spare_part[0])

def Arta_Cement(file_name):
    """
    Translate Arta Cement file names
    """
    #==================================== regex file names type one ============================================================
    signal_name = find_signal_name(file_name)
    # ==========================================================================================================================
    name = re.findall(r"(.*)\s\-\w+.*",signal_name)
    name[0] = name[0].strip()
    point_direction = re.findall(r".*\s.*\-(.*)\s\s\_.*",signal_name)
    # Seperate poit from direction =============================================================================================
    point = str(int(point_direction[0][0:2]))
    direction = point_direction[0][2]
    # ==========================================================================================================================
    spare_part = find_spare(name, point)
    assignment = re.findall(r".*\s(...)\s",signal_name)
    assignment = normalize_assignment(assignment[0])
    date = re.findall(r".*\((.*)\s\d",signal_name)
    hour = re.findall(r".*\(.*\s(.*)\)\s\-",signal_name)
    temp = date[0].split('_')
    date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
    temp = hour[0].split('_')
    hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
    name[0] = name[0].strip()
    date = striptime(date[0])
    #==================================== Return data ==========================================================================
    return make_result_dictionary(name[0] , point, direction, assignment, date, hour[0], spare_part[0])

def Shahrekord_Cement(file_name):
    """
    Translate Shahrekord Cement file names
    """
    #==================================== regex file names type one ============================================================
    signal_name = find_signal_name(file_name)
    # ==========================================================================================================================
    name = re.findall(r"(.+)\..*\s.*\-.*\_.*\s.*",signal_name)
    name[0] = name[0].strip()
    point_direction = re.findall(r".*\s.*\-(.*)\_\s",signal_name)
    # Seperate poit from direction =============================================================================================
    # Find point
    point = re.findall(r'\d+', point_direction[0])[0]
    # Find direction
    direction = re.findall(r'[A-Za-z]+', point_direction[0])[0]
    # ==========================================================================================================================
    spare_part = find_spare(name, point)
    assignment = re.findall(r".*\s(...)\s",signal_name)
    assignment = normalize_assignment(assignment[0])
    date = re.findall(r".*\((.*)\s\d",signal_name)
    hour = re.findall(r".*\(.*\s(.*)\)\s\-",signal_name)
    temp = date[0].split('_')
    date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
    temp = hour[0].split('_')
    hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
    name[0] = name[0].strip()
    date = striptime(date[0])
    # #==================================== Return data ==========================================================================
    return make_result_dictionary(name[0] , point, direction, assignment, date, hour[0], spare_part[0])

def Lar_Cement(file_name):
    """
    Translate Lar Cement file names
    """
    #==================================== regex file names type one ============================================================
    signal_name = find_signal_name(file_name)
    # ==========================================================================================================================
    name = re.findall(r"(.+)\..*\s.*\-.*\_.*\s.*",signal_name)
    name[0] = name[0].strip()
    point_direction = re.findall(r".*\s.*\-(.*)\_\s",signal_name)
    # Seperate poit from direction =============================================================================================
    # Find point
    point = re.findall(r'\d+', point_direction[0])[0]
    # Find direction
    direction = re.findall(r'[A-Za-z]+', point_direction[0])[0]
    # ==========================================================================================================================
    spare_part = re.findall(r".*\s(.*)\-.*\_\s",signal_name)
    assignment = re.findall(r".*\s(...)\s",signal_name)
    assignment = normalize_assignment(assignment[0])
    date = re.findall(r".*\((.*)\s\d",signal_name)
    hour = re.findall(r".*\(.*\s(.*)\)\s\-",signal_name)
    temp = date[0].split('_')
    date[0] = temp[2] + '-' + temp[0] + '-' + temp[1]
    temp = hour[0].split('_')
    hour[0] = temp[0] + ':' + temp[1] + ':' + temp[2]
    name[0] = name[0].strip()
    date = striptime(date[0])
    # # #==================================== Return data ==========================================================================
    return make_result_dictionary(name[0] , point, direction, assignment, date, hour[0], spare_part[0])

# ==============================================================================================================================


# Main
def translator(file_path : str, CompanyName = None):
    """
    This function translated the file names to values we need.\n
    input = {file_path : ADDRESS OF FILE,\n CompanyName : NAME OF THE COMPANY / DEFAULT = Zave Cement}
    """
    # Find right company =======================================================================================================
    try:
        # If CompanyName is not provided
        if CompanyName == None:
            CompanyName = regex(file_path)
        else:
            CompanyName = CompanyName.lower()
            if CompanyName == 'zave cement':
                return Zave_Cement(file_path)
            elif CompanyName == 'arta cement':
                return Arta_Cement(file_path)
            elif CompanyName == 'shahrekord cement':
                return Shahrekord_Cement(file_path)
            elif CompanyName == 'lar cement':
                return Lar_Cement(file_path)
    except:
        logging.warning(f'Pre Processing - Translator: Invalid time signal or dont have required regex to read it.!\n========================================')
        print('====================================================')
        traceback.print_exc()
        print('====================================================')
        return {'name' : 'unknown', 'point' : 'unknown', 'direction' : 'unknown', 'assignment' : 'unknown', 'date' : 'unknown', 'hour' : 'unknown', 'spare_part' : 'unknown'}