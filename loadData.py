import numpy as np
import pandas as pd
import xarray as xr
import dask
import glob


# CONTENTS

# calculate average wind direction
# read data from
# - HOBO weather station @ inlet
# - PROMICE station @ glacier
# - MET weather station @ valley1
# - Statkraft weather station @ valley2 and outlet
# - LiDAR
# - Radiosonde
# - iMet
# - UAV wind
# - Tinytag
# - Humilog



## DEGREES TO RADIANS
def deg2rad(angle:float) -> float:
    import math
    import numpy as np
    angle = np.array([iangle-360 if iangle>=360 else iangle for iangle in list(angle)])
    return angle * math.pi / 180.

## RADIANS TO DEGREES
def rad2deg(anglerad:float) -> float:
    import math
    return anglerad * 180 / math.pi

## CALCULATE WIND DIRECTION AVERAGE
def direction_avg(D0:'np array') -> float:
    """
    Calculate wind direction average.
    angle1 -- list of angles to be averaged.
    return -- average.
    """
    import numpy as np

    ph = D0/180*np.pi
    Ds = np.sin(ph)
    Dc = np.cos(ph)

    wd0 = 180/np.pi*np.arctan2(Ds.mean(),Dc.mean())
    if wd0 < 0:            
        wd0 += 360
    mean_wd = wd0
    return mean_wd  
    

# --------------------------------------------------------------------
# HOBO weather station @ inlet
# --------------------------------------------------------------------

path = 'data/AWS/'
filename = 'HOBO.csv'
url = 'https://www.hobolink.com/p/4717965e3ebf930f87fbbfbe870a4c37#'
file = path+filename
cols = pd.read_csv(file, nrows=1).columns
FF = pd.read_csv(file, usecols=cols[1:])

for col in FF.columns:
    if col[:4] == 'Temp':
        FF = FF.rename(columns={col: 'T'})
    if col[:2] == 'RH':
        FF = FF.rename(columns={col: 'RH'})
    if col[:3] == 'Dew':
        FF = FF.rename(columns={col: 'DP'})
    if col[:8] == 'Wind Dir':
        FF = FF.rename(columns={col: 'WD'})
    if col[:10] == 'Wind Speed':
        FF = FF.rename(columns={col: 'WS'})
    if col[:10] == 'Gust Speed':
        FF = FF.rename(columns={col: 'GS'})
    if col[:40] == 'Solar Radiation (S-LIB 20755830:20748100':
        FF = FF.rename(columns={col: 'SW_out'})
    if col[:40] == 'Solar Radiation (S-LIB 20755830:20748101':
        FF = FF.rename(columns={col: 'SW_in'})
    if col[:4] == 'Rain':
        FF = FF.rename(columns={col: 'PREC'})
        
FF['date'] = pd.to_datetime(FF['Date'])
del FF['Date']

# convert to local hour
FF['date'] += pd.Timedelta(hours=2)

# rotate wind direction by 180 degrees
FF['WD'] -= 180
FF.loc[FF['WD'] < 0, 'WD'] += 360

# calculcate hourly values (from period of samples every 30 min)
#FF['WD'].rolling(min_periods=2, window=2).apply(lambda wd: direction_avg(wd))

# remove precipitation from before rain gauge activation
FF.loc[:np.where(FF['date'] <= pd.Timestamp('2023-09-15 17:00:00'))[0][-1],'PREC'] = np.nan

# identify transition from 30 min to 1 min logging interval
FF['minute'] = (FF['date']).dt.minute.convert_dtypes()
FF['hour'] = (FF['date']).dt.hour.convert_dtypes()
f_2to60 = np.where((FF['minute'] != 0)&(FF['minute'] != 30))[0][0]
f_60first0 = f_2to60+np.where(FF.loc[f_2to60:]['minute']==0)[0][0]
f_2last0 = np.where(FF.loc[:f_2to60-1]['minute']==0)[0][-1]

# calculate hourly precipitation based on 30 min logging interval
FF['PREC_hour'] = FF['PREC'].rolling(min_periods=2, window=2).sum()

# calculate hourly precipitation based on 1 min logging interval
FF['PREC_hour'][f_60first0:] = FF['PREC'][f_60first0:].rolling(min_periods=60, window=60).sum()

# calculate hourly preciptation in transition between 30 min to 1 min logging interval
FF['PREC_hour'][f_60first0] = FF[f_2last0+1:f_60first0+1]['PREC'].sum()

# only keep hourly values at the start of each hour to avoid confusion
FF.loc[FF['minute']!=0, 'PREC_hour'] = np.nan

# calculate daily precipitation
FF['PREC_day'] = np.nan
FF.loc[(FF['PREC_hour']>=0),'PREC_day'] = FF.loc[(FF['PREC_hour']>=0),'PREC_hour'].rolling(min_periods=24, window=24).sum()

# only keep daily values at the start of each day to avoid confusion
FF.loc[FF['hour']!=6, 'PREC_day'] = np.nan


# --------------------------------------------------------------------
# PROMICE weather station @ glacier
# --------------------------------------------------------------------

path = 'data/AWS/'
filename = 'NB_hour.csv'
url = 'https://thredds.geus.dk/thredds/fileServer/aws_l3_station_csv/level_3/UWN/UWN_hour.csv'
file = path+filename
NB = (pd.read_csv(file, sep=','))
NB['date'] = pd.to_datetime(NB['time'])
NB['date'] += pd.Timedelta(minutes=90) # convert to local time and center hourly average
NB = NB.loc[np.where(NB['date'] >= pd.Timestamp('2023-09-10 00:00:00'))[0][0]:].reset_index(drop=True)

# rotate old data by 180 degrees
if (NB['wdir_u'][0] < 200):
    NB.loc[:np.where(NB['date'] <= pd.Timestamp('2023-09-12 17:00:00'))[0][-1],'wdir_u'] += 180
    print ('correcting wind direction')
NB.loc[NB['wdir_u']>360,'wdir_u'] -= 360

# field notes
rotation_dates = [pd.Timestamp('2021-09-02 12:00:00'), pd.Timestamp('2023-09-12 17:00:00')]
rotation_offset = [np.nan, 60]


# --------------------------------------------------------------------
# MET weather station @ valley1 (MG) and mountain (SB)
# --------------------------------------------------------------------

path = 'data/AWS/'
filename = 'MG_hour.csv'
cols = pd.read_csv(path+filename, sep=';', nrows=1).columns
MG = pd.read_csv(path+filename, sep=';', decimal=',', na_values='-', usecols=cols[2:])
MG = MG.rename(columns={MG.columns[0]: 'date', MG.columns[1]: 'PREC', MG.columns[2]: 'T'})
MG['date'] = pd.to_datetime(MG['date'], format='%d.%m.%Y %H:%M')
MG = MG[:-1]

# manually correct from UTC+1 to local summer time
MG['date'] += pd.Timedelta(hours=1)

MG['hour'] = (MG['date']).dt.hour.convert_dtypes()

# calculate daily precipitation
MG['PREC_day'] = np.nan
MG['PREC_day'] = MG['PREC'].rolling(min_periods=24, window=24).sum()

# only keep daily values at the start of each day to avoid confusion
MG.loc[MG['hour']!=6, 'PREC_day'] = np.nan

path = 'data/AWS/'
filename = 'SB_hour.csv'
cols = pd.read_csv(path+filename, sep=';', nrows=1).columns
SB = pd.read_csv(path+filename, sep=';', decimal=',', na_values='-', usecols=cols[2:])
SB = SB.rename(columns={SB.columns[0]: 'date', SB.columns[1]: 'WD', SB.columns[2]: 'WS', SB.columns[3]: 'T'})
SB['date'] = pd.to_datetime(SB['date'], format='%d.%m.%Y %H:%M')
SB = SB[:-1]
SB['date'] += pd.Timedelta(hours=1) # convert to local summer time


# --------------------------------------------------------------------
# Statkraft weather station @ valley2 (BH) and outlet (NV)
# --------------------------------------------------------------------

path = 'data/AWS/'

# data from Steinmannen
#filename = 'SM.xlsx'
#SM = pd.read_excel(path+filename)
#SM = (SM.replace(' ', np.NaN,regex=True))
#SM = SM.rename(columns={'Unnamed: 0': 'date', 'Jost-Steinmann\nen...-T0008A3R\n-0119\n[W/m2]': 'SW_in', 'Jost-Steinmann\nen...-T0014A3K\nI0113\n[Grader]': 'WD', 'Jost-Steinmann\nen...-T0014A3K\nI0113\n[Grader]\norig': 'WD_orig', 'Jost-Steinmann\nen...-T0015A3K\nI0120\n[m/sek]': 'WS', 'Jost-Steinmann\nen...-T0015A3K\nI0120\n[m/sek]\norig': 'WS_orig', 'Jost-Steinmann\nen...-T0017A3K\nI0114\n[Grader C.]': 'T', 'Jost-Steinmann\nen...-T0017A3K\nI0114\n[Grader C.]\norig': 'T_orig'})
#for i in range(len(SM['date'])):
#    try:
#        SM['date'][i] = datetime.strptime(SM['date'][i],'%d/%m/%Y/%H')
#    except:
#        if SM['date'][i][-2:] == '24':
#            SM['date'][i] = SM['date'][i][:-2]+'00' # replacing 24 by 00 for correct time format
#            SM['date'][i] = pd.to_datetime(SM['date'][i],format="%d/%m/%Y/%H")+pd.to_timedelta(1,'d') # adding one day
#        else:
#            SM['date'][i] = pd.to_datetime(SM['date'][i],format="%d/%m/%Y/%H")
#SM['date'] = pd.to_datetime(SM['date'])
#SM = SM.loc[np.where(SM['date'] >= pd.Timestamp('2023-04-19 00:00:00'))[0][0]:].reset_index(drop=True) #### adjust
#
## manually correct from summertime (UTC+2) to UTC
##SM['date']-=pd.Timedelta(hours=2)

# data from Bjørkehaugen
filename = 'Statkraft.xlsx'
BH = pd.read_excel(path+filename, usecols="A:B,G:H,J")
BH = (BH.replace(' ', np.NaN,regex=True))

BH = BH.rename(columns={BH.columns[0]: 'date', BH.columns[1]: 'prec', BH.columns[2]: 'wd', BH.columns[3]: 'ws', BH.columns[4]: 't'})#, 'Jost-Steinmann\nen...-T0008A3R\n-0119\n[W/m2]': 'SW_in', 'Jost-Steinmann\nen...-T0014A3K\nI0113\n[Grader]': 'WD', 'Jost-Steinmann\nen...-T0014A3K\nI0113\n[Grader]\norig': 'WD_orig', 'Jost-Steinmann\nen...-T0015A3K\nI0120\n[m/sek]': 'WS', 'Jost-Steinmann\nen...-T0015A3K\nI0120\n[m/sek]\norig': 'WS_orig', 'Jost-Steinmann\nen...-T0017A3K\nI0114\n[Grader C.]': 'T', 'Jost-Steinmann\nen...-T0017A3K\nI0114\n[Grader C.]\norig': 'T_orig'})
for i in range(len(BH['date'])):
    try:
        BH['date'][i] = datetime.strptime(BH['date'][i],'%d/%m/%Y/%H')
    except:
        if BH['date'][i][-2:] == '24':
            BH['date'][i] = BH['date'][i][:-2]+'00' # replacing 24 by 00 for correct time format
            BH['date'][i] = pd.to_datetime(BH['date'][i],format="%d/%m/%Y/%H")+pd.to_timedelta(1,'d') # adding one day
        else:
            BH['date'][i] = pd.to_datetime(BH['date'][i],format="%d/%m/%Y/%H")
BH['date'] = pd.to_datetime(BH['date'])

# manually correct from summertime (UTC+2) to UTC
#BH['date']-=pd.Timedelta(hours=2)

BH['hour'] = (BH['date']).dt.hour.convert_dtypes()

# calculate daily precipitation
BH['prec_day'] = np.nan
BH['prec_day'] = BH['prec'].rolling(min_periods=24, window=24).sum()

# only keep daily values at the start of each day to avoid confusion
BH.loc[BH['hour']!=6, 'prec_day'] = np.nan

# data from Nigardsvatn
filename = 'Statkraft.xlsx'
NV = pd.read_excel(path+filename, usecols="A,M")
NV = (NV.replace(' ', np.NaN,regex=True))

NV = NV.rename(columns={NV.columns[0]: 'date', NV.columns[1]: 't'})
for i in range(len(NV['date'])):
    try:
        NV['date'][i] = datetime.strptime(NV['date'][i],'%d/%m/%Y/%H')
    except:
        if NV['date'][i][-2:] == '24':
            NV['date'][i] = NV['date'][i][:-2]+'00' # replacing 24 by 00 for correct time format
            NV['date'][i] = pd.to_datetime(NV['date'][i],format="%d/%m/%Y/%H")+pd.to_timedelta(1,'d') # adding one day
        else:
            NV['date'][i] = pd.to_datetime(NV['date'][i],format="%d/%m/%Y/%H")
NV['date'] = pd.to_datetime(NV['date'])

# manually correct from summertime (UTC+2) to UTC
#NV['date']-=pd.Timedelta(hours=2)


# --------------------------------------------------------------------
# LiDAR data
# --------------------------------------------------------------------

dask.config.set({'array.slicing.split_large_chunks': True})

path = 'data/lidar/'
lidar = xr.open_mfdataset(path+'VAD_149_70_v04_2023091*')#,engine='netcdf4')
lidar['time'] = lidar['time'] + np.timedelta64(2,'h')

lidar_ws = np.sqrt(lidar['u'].T**2+lidar['v'].T**2)

lidar_wd = (np.arctan2(lidar['v'].T/lidar_ws,lidar['u'].T/lidar_ws)*180/np.pi)+180
lidar_wd = 90 - lidar_wd
lidar_wd = (lidar_wd%360)




# --------------------------------------------------------------------
# Radiosonde data
# --------------------------------------------------------------------

path = 'data/radiosonde/'
filename = 'balloon2_profile_230912_1313.txt'#balloon2
date = filename[17:23]
#cols = pd.read_csv(path+filename, sep='\tab', nrows=1).columns
RS2 = pd.read_csv(path+filename, sep='\t', decimal='.')#, usecols=[1:])#, userows=rows[1:3])#, na_values='-'
RS2 = RS2.rename(columns={RS2.columns[1]: 'time', RS2.columns[2]: 'p', RS2.columns[3]: 't', RS2.columns[4]: 'rh', RS2.columns[5]: 'ws', RS2.columns[6]: 'wd'})
RS2['date'] = pd.to_datetime(date, format='%y%m%d')+pd.to_timedelta(RS2['time'])
RS2['date']
RS2['date'] += pd.Timedelta(hours=2) # convert to local summer time
RS2 = RS2.drop(['Time [sec]', 'time'], axis=1)

filename = 'balloon4_profile_230913_0719.txt'#balloon2
date = filename[17:23]
#cols = pd.read_csv(path+filename, sep='\tab', nrows=1).columns
RS4 = pd.read_csv(path+filename, sep='\t', decimal='.')#, usecols=[1:])#, userows=rows[1:3])#, na_values='-'
RS4 = RS4.rename(columns={RS4.columns[1]: 'time', RS4.columns[2]: 'p', RS4.columns[3]: 't', RS4.columns[4]: 'rh', RS4.columns[5]: 'ws', RS4.columns[6]: 'wd'})
RS4['date'] = pd.to_datetime(date, format='%y%m%d')+pd.to_timedelta(RS4['time'])
RS4['date']
RS4['date'] += pd.Timedelta(hours=2) # convert to local summer time
RS4 = RS4.drop(['Time [sec]', 'time'], axis=1)

# read gps data from separate file and add to radiosonde file

filename = 'balloon2_202309121139_gps.txt'
date = filename[9:17]
RS2_gps = pd.read_csv(path+filename, sep='\t', decimal='.', encoding='latin-1', usecols=[1,5,6,7], na_values='---')#, userows=rows[1:3])#, na_values='-'
RS2_gps.replace(r'^---\s*$', np.nan, regex=True, inplace=True)
RS2_gps = RS2_gps.rename(columns={RS2_gps.columns[0]: 'time', RS2_gps.columns[1]: 'lon', RS2_gps.columns[2]: 'lat', RS2_gps.columns[3]: 'z'})
RS2_gps['date'] = pd.to_datetime(date, format='%Y%m%d')+pd.to_timedelta(RS2_gps['time'])
RS2_gps['date'] = RS2_gps['date'].dt.round('1s')
RS2_gps['date'] += pd.Timedelta(hours=2) # convert to local summer time
RS2_gps = RS2_gps.drop(['time'], axis=1)
RS2_gps['lon'] = RS2_gps['lon'].astype(float)
RS2_gps['lat'] = RS2_gps['lat'].astype(float)
RS2_gps = RS2_gps.groupby('date').mean().reset_index()
RS2 = pd.merge(RS2, RS2_gps, on='date')

filename = 'balloon4_202309130626_gps.txt'
date = filename[9:17]
RS4_gps = pd.read_csv(path+filename, sep='\t', decimal='.', encoding='latin-1', usecols=[1,5,6,7], na_values='---')#, userows=rows[1:3])#, na_values='-'
RS4_gps.replace(r'^---\s*$', np.nan, regex=True, inplace=True)
RS4_gps = RS4_gps.rename(columns={RS4_gps.columns[0]: 'time', RS4_gps.columns[1]: 'lon', RS4_gps.columns[2]: 'lat', RS4_gps.columns[3]: 'z'})
RS4_gps['date'] = pd.to_datetime(date, format='%Y%m%d')+pd.to_timedelta(RS4_gps['time'])
RS4_gps['date'] = RS4_gps['date'].dt.round('1s')
RS4_gps['date'] += pd.Timedelta(hours=2) # convert to local summer time
RS4_gps = RS4_gps.drop(['time'], axis=1)
RS4_gps['lon'] = RS4_gps['lon'].astype(float)
RS4_gps['lat'] = RS4_gps['lat'].astype(float)
RS4_gps = RS4_gps.groupby('date').mean().reset_index()
RS4 = pd.merge(RS4, RS4_gps, on='date')

# removing first part before launch and gps signal (altitude up and down)
RS2 = RS2.loc[90:].reset_index(drop=True)
RS4 = RS4.loc[236:].reset_index(drop=True)

# calculate potential temperature
RS2['pt'] = (RS2['t']+273.15)*(1013/RS2['p'])**(0.286)
RS4['pt'] = (RS4['t']+273.15)*(1013/RS4['p'])**(0.286)

RS2_ind = np.where(abs(RS2.loc[0,'lat']-RS2['lat'])>0.003)[0][0]
RS4_ind = np.where(abs(RS4.loc[0,'lat']-RS4['lat'])>0.003)[0][0]

# --------------------------------------------------------------------
# iMet data
# --------------------------------------------------------------------

path = 'data/UAV/iMet/'
extension = 'csv'

fns = glob.glob(path+'*.{}'.format(extension))

imets = {}
start = pd.Timestamp('2023-09-12 12:30:00')
end = pd.Timestamp('2023-09-14 16:00:00')

for fn in fns:#['20230914-193303-00061169.csv','20230915-141957-00065717.csv']:#fns:
    if fn != 'UAV/iMet/LATER_20231018-092721-00065717_glacier.csv':
        print (fn)#[11:13]+'-'+fn[-9:-4])
        imet = (pd.read_csv(fn, sep=',', usecols=['XQ-iMet-XQ Pressure','XQ-iMet-XQ Air Temperature','XQ-iMet-XQ Humidity','XQ-iMet-XQ Humidity Temp', 'XQ-iMet-XQ Date', 'XQ-iMet-XQ Time','XQ-iMet-XQ Longitude', 'XQ-iMet-XQ Latitude', 'XQ-iMet-XQ Altitude']))
        imet.columns = imet.columns.str[11:]
        # using only rows where gps signal is acquired
        imet = imet.loc[imet['Latitude']!=0.0]
        imet = imet.loc[(imet['Altitude'] > 280)&(imet['Altitude'] < 1500)]
        try: #if fn == 'UAV/iMet/20230915-141957-00065717_front.csv' or fn == 'UAV/iMet/20230913-141957-00065717_paraglider.csv':
            imet['date'] = pd.to_datetime(imet['Date']+' '+imet['Time'], format='%d/%m/%Y %H:%M:%S')
        except:
            imet['date'] = pd.to_datetime(imet['Date']+' '+imet['Time'], format='%Y/%m/%d %H:%M:%S')
        else:
            imet['date'] = pd.to_datetime(imet['Date']+' '+imet['Time'], format='%d/%m/%Y %H:%M:%S')
        del imet['Date'], imet['Time']
        imet['date'] = pd.to_datetime(imet['date'])
        imet['date'] += pd.Timedelta(hours=2) # convert to local time
        if fn == 'UAV/iMet/20231114-092721-00065717.csv':
            imet_oct = imet
        else:
            imet = imet.loc[(imet['date'] >= start)&(imet['date'] <= end)]
            imets[fn[15:17]+'-'+fn[-9:-4]] = imet


# --------------------------------------------------------------------
# UAV wind data
# --------------------------------------------------------------------

path = 'data/UAV/wind_profiles/'
extension = 'csv'

fns = glob.glob(path+'*.{}'.format(extension))

uav_wind = {}
start = pd.Timestamp('2023-09-12 12:30:00')
end = pd.Timestamp('2023-09-14 16:00:00')

for fn in fns:
    uav = (pd.read_csv(fn, sep=','))#, usecols=['']))
    relevant_rows = uav[uav['Profile ID'].str[0].isin(['1'])]
    if len(relevant_rows) > 0:
        first_index = relevant_rows.index[0]
        last_index = relevant_rows.index[-1]
        uav['date'] = pd.to_datetime(fn[-23:-4], format='%Y-%m-%d_%H-%M-%S')
        uav['date'] += (pd.to_timedelta((uav['Flight time'].str[:2]).astype(int),unit='m')) + (pd.to_timedelta((uav['Flight time'].str[4:6]).astype(int),unit='s'))
    #    uav['date'] += (pd.to_timedelta(2,unit='h'))
        tag = (uav['Profile ID'].iloc[first_index])#[uav['Profile ID'].first_valid_index()])
        uav['Altitude'] = (uav['Altitude'].str[:-2]).astype(float)
        uav['Altitude'] += uav['Start Altitude']
        uav['Wind Direction'] = (uav['Wind Direction'].str[:-1]).astype(float)
        uav['Wind Speed'] = (uav['Wind Speed'].str[:-4]).astype(float)
        # save to dictionary
        uav_wind[tag] = uav.iloc[first_index:last_index].reset_index(drop=True)

# --------------------------------------------------------------------
# Tinytag data
# --------------------------------------------------------------------

global ttFF, ttFF_lon, ttFF_lat, ttFF_hordist

# read tinytag data from forefield
path = 'data/tinytags/FF/'
extension = '.xlsx'
fns = glob.glob(path+'*{}'.format(extension))

ttFF = pd.DataFrame()
for fn in sorted(fns)[:]:
    tt = pd.DataFrame(pd.read_excel(fn, usecols="B,C")[4:])
    ttFF['date_'+fn[12:15]] = tt['Time']#+pd.Timedelta(hours=2) # convert to local summer time
    ttFF['t_'+fn[12:15]] = tt[1]

# read tinytag data from glacier
path = 'data/tinytags/NB/'
extension = '.xlsx'
fns = glob.glob(path+'*{}'.format(extension))

ttNB = pd.DataFrame()
for fn in sorted(fns)[:]:
    tt = pd.DataFrame(pd.read_excel(fn, usecols="B,C")[4:])
    #tt['min'] = tt['Time'].dt.minute.convert_dtypes()
    #tt = tt.loc[tt['min']==0].reset_index() # for easier comparison
    ttNB['date_'+fn[17:20]] = tt['Time']#+pd.Timedelta(hours=2) # convert to local summer time
    ttNB['t_'+fn[17:20]] = tt[1]

# read tinytag data from paraglider
path = 'data/paraglider/'
extension = '.xlsx'
tt = pd.DataFrame(pd.read_excel(path+'hauganosi_temp'+extension)[['Time adjusted to phone time',1,'interpolated altitude']][4:])
tt = tt.rename({'Time adjusted to phone time': 'date', 1: 'temperature', 'interpolated altitude': 'altitude'}, axis='columns')
tt = tt.loc[tt['altitude']>0]
#tt['date'] += pd.Timedelta(hours=2) # convert to local summer time

# tinytag meta data
ttFF_waterdist = [105,199,29,5,20,33,162,27,80] # NB! only horizontal, doesn't account for elevation (HOBO: 77)
ttFF_height = [190,270,205,197,190,205,200,170,170]
ttFF_elevation = [268,280,285,292,279,283,318,332,345] # m a.s.l.
ttFF_hordist = [4290,3510,2730,2450,1900,1360,703,424,487] # distance from front

ttFF_lon = ([7.26476435583597, 7.254492113406914, 7.241973902740535, 7.238890688833559, 7.231686213022527, 7.222722116625928, 7.210152993224942, 7.205165591948366, 7.203816243556431])
ttFF_lat = ([61.658663686221225, 61.663632058887295, 61.66757985115699, 61.66953111473216, 61.67369158695941, 61.67546658819397, 61.67731069319303, 61.67821394773669, 61.67729948065333])

# --------------------------------------------------------------------
# Humilog data
# --------------------------------------------------------------------

path = 'data/humilog/'

fn = 'DK320PDM-22176 25.09.23 10-11-15.ASC'
humilog_lower = pd.read_csv(path+fn, sep='\t', decimal=',', skiprows=5, header=None, encoding='latin-1') #r'\s{2,}'
humilog_lower['date'] = pd.to_datetime(humilog_lower[0]+' '+humilog_lower[1], format='%d.%m.%y %H:%M:%S')
#humilog_lower['date'] += pd.Timedelta(hours=2) # convert to local summer time
del humilog_lower[0],humilog_lower[1],humilog_lower[4]
humilog_lower = humilog_lower.rename(columns={2: 'T', 3: 'RH'})
humilog_lower = humilog_lower.loc[np.where(humilog_lower['date'] >= pd.Timestamp('2023-09-12 18:00:00'))[0][0]:np.where(humilog_lower['date'] >= pd.Timestamp('2023-09-15 14:00:00'))[0][0]].reset_index(drop=True)

fn = 'DK320PDM-22731 29.09.23 12-47-04.ASC'
humilog_upper = pd.read_csv(path+fn, sep='\t', decimal=',', skiprows=5, header=None, encoding='latin-1') #r'\s{2,}'
humilog_upper['date'] = pd.to_datetime(humilog_upper[0]+' '+humilog_upper[1], format='%d.%m.%y %H:%M:%S')
#humilog_upper['date'] += pd.Timedelta(hours=2) # convert to local summer time
del humilog_upper[0],humilog_upper[1],humilog_upper[4]
humilog_upper = humilog_upper.rename(columns={2: 'T', 3: 'RH'})
humilog_upper = humilog_upper.loc[np.where(humilog_upper['date'] >= pd.Timestamp('2023-09-12 18:00:00'))[0][0]:np.where(humilog_upper['date'] >= pd.Timestamp('2023-09-15 14:00:00'))[0][0]].reset_index(drop=True)

humilog_lower['day'] = humilog_lower['date'].dt.day
humilog_lower['hour'] = humilog_lower['date'].dt.hour
humilog_lower_hourly = humilog_lower.groupby(['day','hour']).mean().reset_index(drop=True)
humilog_upper['day'] = humilog_upper['date'].dt.day
humilog_upper['hour'] = humilog_upper['date'].dt.hour
humilog_upper_hourly = humilog_upper.groupby(['day','hour']).mean().reset_index(drop=True)
