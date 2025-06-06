import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
import matplotlib.ticker as tkr
from matplotlib.markers import MarkerStyle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogFormatter 
from matplotlib.colors import ListedColormap

import xarray as xr
import geopandas as gpd
from netCDF4 import Dataset
from datetime import datetime, timedelta
from pyproj import Transformer
from pyproj import Geod
from shapely.geometry import Point, LineString
import rasterio
from rasterio.transform import rowcol
import ast
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import geopy.distance
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import cartopy.crs as ccrs   
import cartopy.feature as cf 

# CONTENTS

# general plot settings
# plot map
# plot wind
# plot temperature
# plot humidity
# plot pressure
# plot radiation
# plot precipitation
# interpolate and plot UAV data
# plot surface data from WRF and observations
# plot WRF cross sections
# plot ERA5


#start = pd.Timestamp('2023-09-11 00:00:00')
#end   = pd.Timestamp('2023-09-21 00:00:00')

start = datetime(2023, 9, 12, 12, 0, 0)
end   = datetime(2023, 9, 14, 18, 0, 0)

IOP_start = datetime(2023, 9, 13, 0, 0, 0)
IOP_end   = datetime(2023, 9, 14, 0, 0, 0)

c = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']



# --------------------------------------------------------------------
# plot map of field campaign
# --------------------------------------------------------------------

def plotMap():
   
    # load map data
    
    terrain = gpd.read_file('data/map/Basisdata_4644_Luster_25833_N50Hoyde_GML.gml', layer='Høydekurve')
    lakes = gpd.read_file('data/map/Basisdata_4644_Luster_25833_N50Arealdekke_GML.gml', layer='Innsjø')
    lakes = lakes.loc[(~np.isnan(lakes['vatnLøpenummer']))&(lakes['høyde']>200)&(lakes['høyde']<400)]
    glaciers = gpd.read_file('data/map/Basisdata_4644_Luster_25833_N50Arealdekke_GML.gml', layer='SnøIsbre')
    rivers = gpd.read_file('data/map/Basisdata_4644_Luster_25833_N50Arealdekke_GML.gml', layer='Elv')
    #labels = gpd.read_file('data/map/Basisdata_4644_Luster_25833_N50Stedsnavn_GML.gml')
    
    terrain_100 = terrain[terrain['høyde'] % 100 == 0]
    terrain_500 = terrain[terrain['høyde'] % 500 == 0]
    glaciers_simp = glaciers.simplify(20, preserve_topology=True)
    glaciers_buffered = glaciers_simp.buffer(.1, resolution=4)
    terrain_glaciers_100 = terrain_100.intersection(glaciers_buffered.union_all())
    terrain_glaciers_500 = terrain_500.intersection(glaciers_buffered.union_all())
    
    world = gpd.read_file('data/map/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
    norway = world[world['ADMIN'] == 'Norway']
    
    # create map
    
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.rcParams.update({'font.size': 24})
    
    terrain_100.to_crs(epsg=4326).plot(ax=ax, lw=1, color='tan', zorder=-1000)
    terrain_500.to_crs(epsg=4326).plot(ax=ax, lw=2, color='tan', zorder=-1000)
    #terrain_500 = terrain_500.to_crs(epsg=4326)
    #contour = ax.contour(X, Y, terrain_500, colors='tan', linewidths=1.8, zorder=-1000)
    #ax.clabel(contour, inline=True, fontsize=8)
    
    glaciers.to_crs(epsg=4326).plot(ax=ax, color='white', edgecolor='k', zorder=-500)
    terrain_glaciers_100.to_crs(epsg=4326).plot(ax=ax, color='skyblue', lw=1, zorder=-100)
    terrain_glaciers_500.to_crs(epsg=4326).plot(ax=ax, color='skyblue', lw=2, zorder=-100)
    rivers.to_crs(epsg=4326).plot(ax=ax, color='skyblue', zorder=-100)
    lakes.to_crs(epsg=4326).plot(ax=ax, color='lightblue', edgecolor='skyblue', zorder=-100)
    ax.set_facecolor('snow')
    
    # manually add some labels
    txt = ax.text(7.1243, 61.6935, ' 1500', c='skyblue', fontsize=9, weight='bold', ha='center', rotation=93)
    txt.set_bbox(dict(facecolor='w', alpha=1, edgecolor='none'))
    txt = ax.text(7.20, 61.7059, '1500', c='skyblue', fontsize=9, weight='bold', ha='center', rotation=-2)
    txt.set_bbox(dict(facecolor='w', alpha=1, edgecolor='none'))
    txt = ax.text(7.2458, 61.68087, '1000', c='tan', fontsize=9, weight='bold', ha='center', rotation=-4)
    txt.set_bbox(dict(facecolor='snow', alpha=1, edgecolor='none'))
    txt = ax.text(7.178, 61.68, ' 1000', c='tan', fontsize=9, weight='bold', ha='center', rotation=101)
    txt.set_bbox(dict(facecolor='snow', alpha=1, edgecolor='none'))
    txt = ax.text(7.28, 61.6648, '500', c='tan', fontsize=9, weight='bold', ha='center', rotation=17)
    txt.set_bbox(dict(facecolor='snow', alpha=1, edgecolor='none'))
    txt = ax.text(7.299, 61.65, '500', c='tan', fontsize=9, weight='bold', ha='center', rotation=-87)
    txt.set_bbox(dict(facecolor='snow', alpha=1, edgecolor='none'))
    txt = ax.text(7.2477, 61.6548, '500', c='tan', fontsize=9, weight='bold', ha='center', rotation=142)
    txt.set_bbox(dict(facecolor='snow', alpha=1, edgecolor='none'))
    
    # add map of Norway
    ax_inset = zoomed_inset_axes(ax, zoom=0.00000004, loc='upper right', #zoom=0.0025
                                 bbox_to_anchor=(.9, 1.126, 0.1, 0.1), #(.947, .9, 0.1, 0.1),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0)
    norway.to_crs(epsg=32633).plot(ax=ax_inset, color='lightgrey')
    norway.to_crs(epsg=32633).boundary.plot(ax=ax_inset, color='black')
    
    transformer = Transformer.from_crs("epsg:4326", "epsg:32633", always_xy=True)
    
    
    # some figure settings
    xmin=7.111416
    xmax=7.31
    ymin=61.638
    ymax=61.713
    sloc = transformer.transform(xmin,ymin)
    ax_inset.scatter(sloc[0], sloc[1], c=c[3], s=100, zorder=200)
    
    alp = 1
    s1 = 600; s2 = 100; s3 = 100; s4 = 800
    m = '^'; m2 = 's'; m3 = 'o'; m4 = 'o'
    lw = 5
    ec = 'k'
    
    start = datetime(2023, 9, 12, 6, 0, 0)
    end   = datetime(2023, 9, 14, 20, 0, 0)
    delay = 0
    
    #for delay in [0]:#[0,2,5,8,11]:
    #    for imet in imets:
    #        if (imets[imet].reset_index()['date'][0] > start)&(imets[imet].reset_index()['date'][0] < end):
    #            imet = imets[imet].loc[(imets[imet]['date'] >= start+pd.Timedelta(hours=delay))&
    #                                   (imets[imet]['date'] <= end+pd.Timedelta(hours=delay))]
    #            upl = ax.scatter(imet['Longitude'],imet['Latitude'],
    #                       s=1,
    #                       vmin=300,vmax=1000,
    #                       c = imet['Altitude'])
    #ax.plot((),(),c='y',label='UAV')
    #plt.colorbar(upl)
    
       
    # location of AWS
    #ax.scatter(7.13220410,        61.67717908,        c=c[3], alpha=alp, ec=ec, marker=m, s=s1, label='AWS$_{plateau}$',   zorder=2)
    ax.scatter(7.197794684331172, 61.686051540061946, c=c[9], alpha=alp, ec=ec, marker=m, s=s1, label='AWS$_{glacier}$',   zorder=2)
    ax.scatter(7.211611010507808, 61.675952354678884, c=c[1], alpha=alp, ec=ec, marker=m, s=s1, label='AWS$_{inlet}$', zorder=2)
    ax.scatter(7.2415509,         61.6672661,         c=c[6], alpha=alp, ec=ec, marker=m, s=s1, label='AWS$_{outlet}$', zorder=2)
    ax.scatter(7.275990675443110, 61.659358589432706, c=c[2], alpha=alp, ec=ec, marker=m, s=s1, label='AWS$_{valley1}$',   zorder=3)
    ax.scatter(7.27426310,        61.66016865,        c=c[5], alpha=alp, ec=ec, marker=m, s=s1, label='AWS$_{valley2}$',   zorder=2)
    #ax.legend(loc='lower left')
    
    handles1, labels1 = ax.get_legend_handles_labels()
    #legend1 = ax.legend(handles1[:6], labels1[:6], bbox_to_anchor=(0.291, 0), loc='lower left')
    legend1 = ax.legend(handles1[:6], labels1[:6], bbox_to_anchor=(0.371, 0), loc='lower left')
    
    # location of tinytags
    vmin = -580
    vmax = 4290
    for i in range(len(ttFF.columns[0::2])):
        if i in [2,3,7]:
            pass
        else:
            ax.scatter(ttFF_lon[i], ttFF_lat[i], c=ttFF_hordist[i], vmin=vmin,vmax=vmax,
                        alpha=alp, ec=ec, marker=m2, s=s2, zorder=10)#, label=f'FF{i+1}') #c=cm.cool((8-i)/8)
    
    #    ax2.scatter(7.195903090958175, 61.68563814017413,   c=cm.cool(.99), alpha=alp, ec=ec, marker=m3, s=s3, label='NB1', zorder=3)
    #    ax2.scatter(7.196770590996879, 61.685596262391066,  c=cm.cool(.67), alpha=alp, ec=ec, marker=m3, s=s3, label='NB2', zorder=3)
    #    ax2.scatter(7.19842560429481,  61.68617729032908,   c=cm.cool(.33), alpha=alp, ec=ec, marker=m3, s=s3, label='NB3', zorder=3)
    #    ax2.scatter(7.199162929886781, 61.68682800158401,   c=cm.cool(0), alpha=alp, ec=ec, marker=m3, s=s3, label='NB4', zorder=3)
    
    # location of other instruments
    #ax.plot([7.2423,7.2110,7.1993,7.1978,7.1870], [61.6674,61.6759,61.6805,61.6858,61.6912], c=c[8], ls='--', lw=3, zorder=0)
    ax.scatter(7.2423, 61.6674, alpha=alp, ec=c[3], fc='none', lw=lw*.6, marker=m4, s=s4, zorder=1)
    ax.scatter(7.2110, 61.6759, alpha=alp, ec=c[3], fc='none', lw=lw*.6, marker=m4, s=s4, zorder=1)
    ax.scatter(7.1993, 61.6805, alpha=alp, ec=c[3], fc=c[7],   lw=lw*.6, marker=m4, s=s4, zorder=1)
    ax.scatter(7.1978, 61.6856, alpha=alp, ec=c[3], fc='none', lw=lw*.6, marker=m4, s=s4, zorder=1)
    #ax.scatter(7.1870, 61.6912, alpha=alp, ec=c[8], fc='none', lw=lw, marker=m4, s=s4, zorder=1)
    ax.scatter(7.232, 61.672, alpha=alp, ec=c[3], fc='none', lw=lw*.6, ls=':', marker=m4, s=1500, zorder=1)
    
    ax.scatter(7.220021783037555, 61.675215447369474, alpha=alp, ec=ec, fc=c[4], marker='*', s=1000, label='LiDAR')#, zorder=20)
    ax.scatter((),(), alpha=alp, ec='None', fc=c[7], marker=m4, s=s4-200, label='Radiosonde')
    ax.scatter((),(), alpha=alp, ec=c[3], fc='none', lw=lw*.6, marker=m4, s=s4, zorder=0, label='UAV')
    ax.scatter((),(), alpha=alp, ec=c[3], fc='none', lw=lw*.6, ls=':', marker=m4, s=600, zorder=1, label='Paraglider') # (part of flight below 1000 m)
    ax.scatter(7.198646470752097, 61.683231702332485, alpha=alp, c=c[8], ec='k', marker='D', s=s2, zorder=1, label='Humilog$_{upper}$')
    ax.scatter(7.198529491371879, 61.6814934541163, alpha=alp, c='grey', ec='k', marker='D', s=s2, zorder=1, label='Humilog$_{lower}$')
    ax.scatter((), (), c='k', alpha=alp, ec=ec, marker=m2, s=s2, label='Tinytags')
    #ax.legend(loc='upper left')
    
    # location of wrf cross section
    ax.plot((7.133056640625, 7.302520751953125), (61.711647033691406, 61.639015197753906), color=c[7], lw=2, ls='--', zorder=-3, label='WRF cross section')
    
    handles2, labels2 = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles2[5:], labels2[5:], bbox_to_anchor=(0, 0), loc='lower left')
    fig.add_artist(legend1)
    fig.add_artist(legend2)
    
    dl = .094
    dla = .037
    rect = patches.Rectangle((7.19052+dl, 61.6397+dla), 7.21354-7.19052, 61.6428-61.6397, linewidth=1, edgecolor='w', facecolor='w')
    ax.add_patch(rect)
    ax.plot([7.19252+dl,7.21154+dl],[61.6405+dla,61.6405+dla],'k')
    ax.plot([7.19252+dl,7.19252+dl],[61.6405+dla,61.642+dla],'k')
    ax.plot([7.21154+dl,7.21154+dl],[61.6405+dla,61.642+dla],'k')
    ax.text(7.1957+dl, 61.641+dla, ' 1 km', fontsize=18)
    
    ax.plot([7.276,7.285],[61.643,61.6407],'k')
    ax.scatter(7.285, 61.641, c='k', alpha=alp, ec=ec, marker=m, s=s1, zorder=2)
    ax.text(7.27, 61.644, 'AWS$_{mountain}$', fontsize=22, ha='center')
    
    ax.text(7.17, 61.6915, 'Nigardsbreen', fontsize=22, ha='center', rotation=-10)
    ax.plot([7.228,7.226],[61.676,61.674],'k')
    ax.text(7.23, 61.6735, 'Nigardsbrevatnet', fontsize=22, ha='center', rotation=-20)
    ax.text(7.245, 61.655, 'Mjølverdalen', fontsize=22, ha='center', rotation=-42)
    
    #ax.text(7.121, 61.6835, 'Jostedalsbreen', fontsize=22, ha='center', rotation=90)
    #ax.annotate('Jostedalsbreen', fontsize=22, xy=(7.115, 61.692), xytext=(7.116, 61.6865), arrowprops=dict(arrowstyle='-|>', facecolor='k'))
    
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel('longitude ($\u00b0$)')
    ax.set_ylabel('latitude ($\u00b0$)')
    
    xmin, ymin = transformer.transform(4.5, 57.2)
    xmax, ymax = transformer.transform(34.5, 71.2)
    ax_inset.set_xlim(xmin, xmax)
    ax_inset.set_ylim(ymin, ymax)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    
    ax.set_xticks(np.arange(7.12,7.31,0.03))
    
    ax.arrow(7.12, 61.7087, 0, 0.0029, width = .0001, length_includes_head=True,
              head_width=0.008, head_length=0.005, overhang=.4, facecolor='k')

    #plt.tight_layout()
    plt.savefig('plots/map_campaign_terrain.pdf', format='pdf')
    plt.show()

# --------------------------------------------------------------------
# plot wind
# --------------------------------------------------------------------


# wind evolution -----------------------------------------------------

def plotWindEvolution():

    plt.rcParams.update({'font.size': 22})
    
    fig,(ax) = plt.subplots(1,figsize=(20,12),dpi=300)
    
    ax.axhline(0.5,c='grey',ls=':')
    ax.plot(SB['date'],SB['WS'],    '-',c='k', label='mountain')
    ax.plot(NB['date'],NB['wspd_u'],'-',c=c[9], label='glacier')
    ax.plot(FF['date'],FF['WS'],    '-',c=c[1], label='inlet', zorder=-10)
    ax.plot(BH['date'],BH['ws'],    '-',c=c[5], label='valley2', zorder=-10)
    ax.axvline(np.datetime64('2023-09-12T17:00:00'),c=c[9],ls='--') # rotated station
    ax.set_ylabel('wind speed (m s$^{-1}$)')
    
    ax2 = ax.twinx()
    windy_days = SB[SB['WS'] >= .5]
    ax2.plot(windy_days['date'],windy_days['WD'], 'o', c='k')
    windy_days = NB[NB['wspd_u'] >= .5]
    ax2.plot(windy_days['date'],windy_days['wdir_u'], 'o', c=c[9])
    windy_days = FF[FF['WS'] >= .5]
    ax2.plot(windy_days['date'],windy_days['WD'], 'o', c=c[1], zorder=-10)
    windy_days = BH[BH['ws'] >= .5]
    ax2.plot(windy_days['date'],windy_days['wd'], 'o', c=c[5])
    ax2.axvline(np.datetime64('2023-09-12T17:00:00'),c=c[9],ls='--') # rotated station
    
    ax2.set_ylabel('wind direction ($\u00b0$)', rotation=270, labelpad=25) # (\u00b0)
    ax2.set_ylim(-6,366)
    ax2.set_yticks(np.arange(0,361,90))
    ax2.set_yticklabels(['N','E','S','W','N'])
    ax.set_yticks(np.arange(0,13,3))
    
    ax2.plot((), (), '-', c='grey', label='wind speed')
    ax2.plot((), (), 'o', c='grey', label='wind direction')
    
    ax.axvspan(xmin=IOP_start, xmax=IOP_end, ymin=0, ymax=1, facecolor='grey', alpha=0.1)
    ax2.axvspan(xmin=IOP_start, xmax=IOP_end, ymin=0, ymax=1, facecolor='grey', alpha=0.1)
    #ax.set_xlim(xmin=start,xmax=end)
    ax.set_xlim(xmin=datetime(2023, 9, 13, 0, 0, 0), xmax=datetime(2023, 9, 15, 0, 0, 0))
    ax.set_xlabel('local time')
    #ax2.set_xlim(xmin=start,xmax=end)
    ax2.set_xlim(xmin=datetime(2023, 9, 13, 0, 0, 0), xmax=datetime(2023, 9, 15, 0, 0, 0))
    ax2.set_xlabel('local time')
    ax.set_ylim(-.2,12.2)
    fig.autofmt_xdate(rotation=45)
    legend = ax.legend(loc=6, bbox_to_anchor=(0.0,0.618))
    legend2 = ax2.legend(loc=6, bbox_to_anchor=(0.0,0.43))
    legend.get_frame().set_alpha(1)
    legend2.get_frame().set_alpha(.65)
    ax.grid()
    ax2.grid()
    plt.savefig('plots/wind.pdf', format='pdf')
    
    plt.show()

# vertical wind profiles ---------------------------------------------

def plotVerticalWindProfiles():

    plt.rcParams.update({'font.size': 24})
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24*.9,15*.9))
    
    h_lidar = 284
    
    alpha = .1
    ms = 3
    m = '*'
    
    # radiosonde
    for ws, wd, z in zip(RS4['ws'], RS4['wd'], RS4['z']):
        if 270 < wd < 360 and ws > 0.5:
            axes[0,0].plot(ws, z, '.', ms=ms*1.5, c=c[7])
        else:
            axes[0,0].plot(ws, z, '.', ms=ms*1.5, c=c[7], alpha=alpha*2)
    
    # wrf
    colours = [c[9],c[7],c[1],c[6]]
    for i,h in enumerate(['07','12','15','18']):
        for j,loc in enumerate(['NB', 'front', 'FF', 'NV']):
            if loc == 'NB':
                dz = 171
            elif loc == 'front':
                dz = 190
            elif loc == 'FF':
                dz = 265
            elif loc =='NV':
                dz = 158
            height = np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/height_{loc}.npy')-dz
            #print (loc, height)
            #height -= dz#100
            #print (height)
            mask = (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_13-{h}.npy') > 270) & \
            (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy') > 1)
            if loc == 'FF' or loc == 'NV':
                mask = (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_13-{h}.npy') > 247.5) & \
                (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_13-{h}.npy') < 337.5) & \
                (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy') > 1)
            axes[0,i].plot(np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy'), 
                           height, lw=3, c=colours[j], alpha=alpha)
            axes[0,i].scatter(np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy')[mask], 
                              height[mask], marker='o', lw=1.5, c='none', ec=colours[j])
            axes[0,i].scatter(np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy')[~mask], 
                              height[~mask], marker='o', lw=1.5, c='none', ec=colours[j], alpha=alpha)
            if h != '18':
                mask = (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_14-{h}.npy') > 270) & \
                (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy') > 1)
                if loc == 'FF' or loc == 'NV':
                    mask = (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_14-{h}.npy') > 247.5) & \
                    (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_14-{h}.npy') < 337.5) & \
                    (np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy') > 1)
                axes[1,i].plot(np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy'), 
                               height, lw=3, c=colours[j], alpha=alpha)
                axes[1,i].scatter(np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy')[mask], 
                                  height[mask], marker='o', lw=1.5, c='none', ec=colours[j])
                axes[1,i].scatter(np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy')[~mask], 
                                  height[~mask], marker='o', lw=1.5, c='none', ec=colours[j], alpha=alpha)
    
    
    # UAV
    prof_glacier = [['13-07-Dm', '13-12-Dm', '13-15-Dm', '13-18-Dm'], ['14-07-Dm', '14-12-Dm', '14-15-Dm']]
    prof_front   = [['13-07-Cm', '13-12-Cm', '13-15-Cm', '13-18-Cm'], ['14-07-Cm', '14-12-Cm', '14-15-Cm']]
    prof_inlet   = [[], ['14-07-Bm', '14-12-Bm', '14-15-Bm']]
    prof_outlet   = [['13-07-Am', '13-12-Am', '13-15-Am', '13-18-Am'], ['14-07-Am', '14-12-Am', '14-15-Am']]
    
    for j in range(2):
        for i,p in enumerate(prof_glacier[j]):
            for ws, wd, z in zip(uav_wind[p]['Wind Speed'], uav_wind[p]['Wind Direction'], uav_wind[p]['Altitude']):
                if 270 < wd < 360 and ws > 0.5:
                    axes[j,i].plot(ws, z, 'o', c=c[9])
                    #print ('glac', p, ws, z)
                else:
                    axes[j,i].plot(ws, z, 'o', c=c[9], alpha=alpha)
        for i,p in enumerate(prof_front[j]):
            for ws, wd, z in zip(uav_wind[p]['Wind Speed'], uav_wind[p]['Wind Direction'], uav_wind[p]['Altitude']):
                if 270 < wd < 360 and ws > 0.5:
                    axes[j,i].plot(ws, z, 'o', c=c[7])
                    #print ('front', p, ws, z)
                else:
                    axes[j,i].plot(ws, z, 'o', c=c[7], alpha=alpha)
        if j==1:
            for i,p in enumerate(prof_inlet[j]):
                for ws, wd, z in zip(uav_wind[p]['Wind Speed'], uav_wind[p]['Wind Direction'], uav_wind[p]['Altitude']):
                    if 247.5 < wd < 337.5 and ws > 0.5:
                        axes[j,i].plot(ws, z, 'o', c=c[1])
                        #print ('inlet', p, ws, z)
                    else:
                        axes[j,i].plot(ws, z, 'o', c=c[1], alpha=alpha)  
        for i,p in enumerate(prof_outlet[j]):
            for ws, wd, z in zip(uav_wind[p]['Wind Speed'], uav_wind[p]['Wind Direction'], uav_wind[p]['Altitude']):
                if 247.5 < wd < 337.5 and ws > 0.5:
                    axes[j,i].plot(ws, z, 'o', c=c[6])
                    #print ('outlet', p, ws, z)
                else:
                    axes[j,i].plot(ws, z, 'o', c=c[6], alpha=alpha)
    
    # lidar
    lidar_sel = lidar.sel(time=slice('2023-09-13T06:45:00', '2023-09-13T07:15:00'))
    #axes[0,0].plot(lidar_sel['wspeed'], h_lidar+lidar_sel['height'].broadcast_like(lidar_sel['wspeed']), m, c=c[4])
    #for ax in axes[0, :]:
    #    ax.plot(lidar_sel['wspeed'], h_lidar+lidar_sel['height'].broadcast_like(lidar_sel['wspeed']), m, c=c[4], ms=ms, alpha=alpha)
    for k,z in enumerate(lidar_sel['height']):
        for ws, wd in zip(lidar_sel['wspeed'][:,k], lidar_sel['wdir'][:,k]):
            if 247.5 < wd < 337.5 and ws > 0.5:
                axes[0,0].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3)
            else:
                axes[0,0].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3, alpha=alpha*2)
    lidar_sel = lidar.sel(time=slice('2023-09-13T11:45:00', '2023-09-13T12:15:00'))
    #for ax in axes[0, :]:
    #    ax.plot(lidar_sel['wspeed'], h_lidar+lidar_sel['height'].broadcast_like(lidar_sel['wspeed']), m, c=c[4], ms=ms, alpha=alpha)
    for k,z in enumerate(lidar_sel['height']):
        for ws, wd in zip(lidar_sel['wspeed'][:,k], lidar_sel['wdir'][:,k]):
            if 247.5 < wd < 337.5 and ws > 0.5:
                axes[0,1].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3)
            else:
                axes[0,1].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3, alpha=alpha*2)
    lidar_sel = lidar.sel(time=slice('2023-09-13T14:45:00', '2023-09-13T15:15:00'))
    #for ax in axes[0, :]:
    #    ax.plot(lidar_sel['wspeed'], h_lidar+lidar_sel['height'].broadcast_like(lidar_sel['wspeed']), m, c=c[4], ms=ms, alpha=alpha)
    for k,z in enumerate(lidar_sel['height']):
        for ws, wd in zip(lidar_sel['wspeed'][:,k], lidar_sel['wdir'][:,k]):
            if 247.5 < wd < 337.5 and ws > 0.5:
                axes[0,2].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3)
            else:
                axes[0,2].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3, alpha=alpha*2)
    lidar_sel = lidar.sel(time=slice('2023-09-13T17:45:00', '2023-09-13T18:15:00'))
    #for ax in axes[0, :]:
    #    ax.plot(lidar_sel['wspeed'], h_lidar+lidar_sel['height'].broadcast_like(lidar_sel['wspeed']), m, c=c[4], ms=ms, alpha=alpha)
    for k,z in enumerate(lidar_sel['height']):
        for ws, wd in zip(lidar_sel['wspeed'][:,k], lidar_sel['wdir'][:,k]):
            if 247.5 < wd < 337.5 and ws > 0.5:
                axes[0,3].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3)
            else:
                axes[0,3].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3, alpha=alpha*2)
    lidar_sel = lidar.sel(time=slice('2023-09-14T06:45:00', '2023-09-14T07:15:00'))
    #for ax in axes[1, :]:
    #    ax.plot(lidar_sel['wspeed'], h_lidar+lidar_sel['height'].broadcast_like(lidar_sel['wspeed']), m, c=c[4], ms=ms, alpha=alpha)
    for k,z in enumerate(lidar_sel['height']):
        for ws, wd in zip(lidar_sel['wspeed'][:,k], lidar_sel['wdir'][:,k]):
            if 247.5 < wd < 337.5 and ws > 0.5:
                axes[1,0].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3)
            else:
                axes[1,0].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3, alpha=alpha*2)
    lidar_sel = lidar.sel(time=slice('2023-09-14T11:45:00', '2023-09-14T12:15:00'))
    #for ax in axes[1, :]:
    #    ax.plot(lidar_sel['wspeed'], h_lidar+lidar_sel['height'].broadcast_like(lidar_sel['wspeed']), m, c=c[4], ms=ms, alpha=alpha)
    for k,z in enumerate(lidar_sel['height']):
        for ws, wd in zip(lidar_sel['wspeed'][:,k], lidar_sel['wdir'][:,k]):
            if 247.5 < wd < 337.5 and ws > 0.5:
                axes[1,1].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3)
            else:
                axes[1,1].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3, alpha=alpha*2)
    lidar_sel = lidar.sel(time=slice('2023-09-14T14:45:00', '2023-09-14T15:15:00'))
    #for ax in axes[1, :]:
    #    ax.plot(lidar_sel['wspeed'], h_lidar+lidar_sel['height'].broadcast_like(lidar_sel['wspeed']), m, c=c[4], ms=ms, alpha=alpha)
    for k,z in enumerate(lidar_sel['height']):
        for ws, wd in zip(lidar_sel['wspeed'][:,k], lidar_sel['wdir'][:,k]):
            if 247.5 < wd < 337.5 and ws > 0.5:
                axes[1,2].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3) #mfc='none', 
            else:
                axes[1,2].plot(ws, h_lidar+z, m, c=c[4], ms=ms*3, alpha=alpha*2) #mfc='none', 
    
    # AWS
    for i,h in enumerate([7,12,15,18]):
        if 270 <= SB.loc[SB['date']==datetime(2023,9,13,h,0,0),'WD'].values[0] <= 360 and \
        SB.loc[SB['date']==datetime(2023,9,13,h,0,0),'WS'].values[0] > 0.5:
            axes[0,i].plot(SB.loc[SB['date']==datetime(2023,9,13,h,0,0),'WS'], 1566, '^', c='k', ms=ms*4)
        else:
            axes[0,i].plot(SB.loc[SB['date']==datetime(2023,9,13,h,0,0),'WS'], 1566, '^', c='k', ms=ms*4, alpha=alpha*3)
        if 270 <= NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wdir_u'].values[0] <= 360 and \
        NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'].values[0] > 0.5:
            #for ax in axes[0, :]:
            #    ax.plot(NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'], 550, 'X', c=c[9], ms=ms*4, alpha=alpha*1.5)
            axes[0,i].plot(NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'], 550, '^', c=c[9], ms=ms*4)
        else:
            #for ax in axes[0, :]:
            #    ax.plot(NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'], 550, 'x', c=c[9], ms=ms*4, alpha=alpha*1.5)
            axes[0,i].plot(NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'], 550, '^', c=c[9], ms=ms*4, alpha=alpha*1.5)
        if 247.5 <= FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WD'].values[0] <= 337.5 and \
        FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'].values[0] > 0.5:
            #for ax in axes[0, :]:
            #    ax.plot(FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'], 277, 'X', c=c[1], ms=ms*4, alpha=alpha*1.5)
            axes[0,i].plot(FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'], 277, '^', c=c[1], ms=ms*4)
        else:
            #for ax in axes[0, :]:
            #    ax.plot(FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'], 277, 'x', c=c[1], ms=ms*4, alpha=alpha*1.5)
            axes[0,i].plot(FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'], 277, '^', c=c[1], ms=ms*4, alpha=alpha*1.5)
    for i,h in enumerate([7,12,15]):
        if 270 <= SB.loc[SB['date']==datetime(2023,9,14,h,0,0),'WD'].values[0] <= 360 and \
        SB.loc[SB['date']==datetime(2023,9,14,h,0,0),'WS'].values[0] > 0.5:
            axes[1,i].plot(SB.loc[SB['date']==datetime(2023,9,14,h,0,0),'WS'], 1566, '^', c='k', ms=ms*4)
        else:
            axes[1,i].plot(SB.loc[SB['date']==datetime(2023,9,14,h,0,0),'WS'], 1566, '^', c='k', ms=ms*4, alpha=alpha*3)
        if 270 <= NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wdir_u'].values[0] <= 360 and \
        NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'].values[0] > 0.5:
            #for ax in axes[1, :]:
            #    ax.plot(NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'], 550, 'X', c=c[9], ms=ms*4, alpha=alpha*1.5)
            axes[1,i].plot(NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'], 550, '^', c=c[9], ms=ms*4)
        else:
            #for ax in axes[1, :]:
            #    ax.plot(NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'], 550, 'x', c=c[9], ms=ms*4, alpha=alpha*1.5)
            axes[1,i].plot(NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'], 550, '^', c=c[9], ms=ms*4, alpha=alpha*1.5)
        if 247.5 <= FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WD'].values[0] <= 337.5 and \
        FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'].values[0] > 0.5:
            #for ax in axes[1, :]:
            #    ax.plot(FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'], 277, 'X', c=c[1], ms=ms*4, alpha=alpha*1.5)
            axes[1,i].plot(FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'], 277, '^', c=c[1], ms=ms*4)
        else:
            #for ax in axes[1, :]:
            #    ax.plot(FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'], 277, 'x', c=c[1], ms=ms*4, alpha=alpha*1.5)
            axes[1,i].plot(FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'], 277, '^', c=c[1], ms=ms*4, alpha=alpha*1.5)
        
    for ax in axes.flat:
        ax.set_xlim(-1, 14)
        ax.set_ylim(210, 2100) # 250, 700) #
        ax.set_xticks([0,4,8,12])
    for row in range(axes.shape[0] - 1):  # Iterate over all rows except the last
        for ax in axes[row, :-1]:           # Iterate over all columns in the given row
            ax.set_xticklabels([])
    for col in range(1, axes.shape[1]):  # Iterate over all columns except the first
        for ax in axes[:, col]:           # Iterate over all rows in the given column
            ax.set_yticklabels([])
    for ax in axes[-1, :]:
        ax.set_xlabel('wind speed (m s$^{-1}$)')
    axes[0,3].set_xlabel('wind speed (m s$^{-1}$)')
    for ax in axes[:, 0]:
        ax.set_ylabel('altitude (m a.s.l.)')
    
    
    legend_ax = fig.add_axes([0.76,0.05,0.23,0.45])#0.775, 0.205, 0.1, 0.15]) # x, y, width, height (in figure coords)
    legend_ax.axis('off')  # Turn off the axis
    
    legend_ax.plot((), (), 's', c='k', label='mountain')
    legend_ax.plot((), (), 's', c=c[9], label='glacier')
    legend_ax.plot((), (), 's', c=c[7], label='front')
    legend_ax.plot((), (), 's', c=c[1], label='inlet')
    legend_ax.plot((), (), 's', c=c[6], label='outlet')
    
    #legend_ax.plot((), (), '-', c='w', alpha=0, label=' ')
    legend_ax.plot((), (), m, ms=ms*3, c=c[4], label='LiDAR')
    legend_ax.plot((), (), '.', ms=ms*1.5, c=c[7], label='RS')
    legend_ax.plot((), (), 'o', c='lightgrey', label='UAV')
    legend_ax.plot((), (), '^', c='lightgrey', ms=ms*4, label='AWS')
    legend_ax.plot((), (), marker='o', c='lightgrey', markerfacecolor='w', lw=1.5, label='WRF')
    
    
    handles1, labels1 = legend_ax.get_legend_handles_labels()
    legend1 = legend_ax.legend(handles1[:], labels1[:], bbox_to_anchor=(0, 0.25), ncol=2, markerscale=1.3, columnspacing=0.8, labelspacing=0.4, handlelength=1.5, borderpad=0.3, loc='lower left')  
    
    fig.add_artist(legend1)
    
    
    axes[0,0].set_title('a)   13 Sept. 07:00 LT  ')
    axes[0,1].set_title('b)   13 Sept. 12:00 LT  ')
    axes[0,2].set_title('c)   13 Sept. 15:00 LT  ')
    axes[0,3].set_title('d)   13 Sept. 18:00 LT  ')
    axes[1,0].set_title('e)   14 Sept. 07:00 LT  ')
    axes[1,1].set_title('f)   14 Sept. 12:00 LT  ')
    axes[1,2].set_title('g)   14 Sept. 15:00 LT  ')
    
    fig.delaxes(axes[1][3])
    
    plt.tight_layout()
    plt.savefig('plots/wind_profiles.pdf', format='pdf')


# wind rose ----------------------------------------------------------

#im = plt.imread('data/map/2022-08-31-00_00_2022-08-31-23_59_Sentinel-2_L1C_True_color_Jostedalen.png')
#
#xmin=7.140083
#xmax=7.422981
#ymin=61.57961
#ymax=61.705173
#
#fig,ax = plt.subplots(1,figsize=(20,12))#706/norm,569/norm))
#plt.imshow(im, extent=[xmin,xmax,ymin,ymax], aspect='auto')
#plt.rcParams.update({'font.size': 24})
#
#cmap = cm.get_cmap('viridis', 10)
##norm = colors.BoundaryNorm(np.arange(.05, .36, .05), cmap.N)
#
##cm = ax.scatter((lon[start:end]), (lat[start:end]), c=(albedo[start:end]), s=30, cmap=cmap, norm=norm)
##fig.colorbar(cm, ax=ax)#, aspect=40)
#
#AWS_lon = [7.13220410,7.197794684331172,7.211611010507808,7.2415509,7.275990675443110,7.27426310,7.3892]
#AWS_lat = [61.67717908,61.686051540061946,61.675952354678884,61.6672661,61.659358589432706,61.66016865,61.5972]
#AWS_col = [c[3],c[9],c[1],c[6],c[2],c[5],'k']
#AWS_lab = ['AWS$_{plateau}$','AWS$_{glacier}$','AWS$_{inlet}$','AWS$_{outlet}$','AWS$_{valley1}$','AWS$_{valley2}$','AWS$_{mountain}$']
#
##for lon,lat,col,lab in zip(AWS_lon,AWS_lat,AWS_col,AWS_lab):
##    ax.scatter(lon, lat, c=col, alpha=alp, ec=ec, marker=m, s=s1, label=lab, zorder=2)
##ax.scatter(7.220021783037555, 61.675215447369474, c=c[4], alpha=alp, ec=ec, marker='*', s=1000, label='LiDAR', zorder=2)
#
#wrs = pd.Timestamp('2023-09-13 00:00:00')
#wre = pd.Timestamp('2023-09-15 00:00:00')
#wdstr = ['nan','wdir_u','WD','nan','nan','wd','WD']
#wsstr = ['nan','wspd_u','WS','nan','nan','ws','WS']
#amlos = AWS_lon
#amlas = AWS_lat
#wrlos = [7.133216813444359,7.214,7.185,7.242,7.276,7.271,7.3892]
#wrlas = [61.693,61.693,61.670,61.667,61.659,61.670,61.5972]
#locs = [10,10,10,10,10,10,10]
#
#asp=2
#for wrlo,wrla,lo,la,col,l,aws,wdstr,wsstr in zip(wrlos,wrlas,amlos,amlas,AWS_col,locs,[SM, NB, FF, NV, MG, BH, SB], wdstr, wsstr):
#    if (wdstr != 'nan'): # excluding SM and MG with unavailable wind data
#        print (l)
#        wd = aws[np.where(aws['date'] >= wrs)[0][0]:np.where(aws['date'] <= wre)[0][-1]][wdstr]
#        ws = aws[np.where(aws['date'] >= wrs)[0][0]:np.where(aws['date'] <= wre)[0][-1]][wsstr]
#        wrax = inset_axes(ax, width=asp, height=asp, loc=l, 
#                          bbox_to_anchor=(lo, la), bbox_transform=ax.transData,
#                          axes_class=WindroseAxes)
#        wrax.set_facecolor((1,1,1,.2))
#        wrax.bar(wd, ws, normed=True, bins=np.arange(0, 7, 1))#, cmap='viridis')
#        wrax.tick_params(labelleft=False, labelbottom=False)#True)
#        #wrax.set_yticks(np.arange(0,71,20))
#        if col == c[9]:
#            wrax.legend(loc=6,bbox_to_anchor=(5., .3), fontsize=18, title='wind speed (m/s)')
#
##        ax.plot([lo,wrlo],[la,wrla], c=col, ls='--', lw=3)
#
#ax.plot([7.24252,7.26154],[61.696,61.696],'k')
#ax.plot([7.24252,7.24252],[61.696,61.6975],'k')
#ax.plot([7.26154,7.26154],[61.696,61.6975],'k')
#ax.text(7.2445,61.697,'1 km')
#
#ax.set_title(f'13-14 Sep')
#ax.set_xlabel('longitude')
#ax.set_ylabel('latitude')
#ax.set_xlim(xmin,xmax)
#ax.set_ylim(ymin,ymax)
#
#plt.savefig('plots/plots_campaign/windrose_map_13-14.png', format='png')
#plt.show()




# --------------------------------------------------------------------
# plot temperature
# --------------------------------------------------------------------

# cold air pool context ----------------------------------------------

def plotColdAirPools():
        
    plt.rcParams.update({'font.size': 22})
    
    fig,(ax2,ax) = plt.subplots(2,figsize=(15,17), gridspec_kw={'height_ratios': [1,5]}, dpi=80)
    
    lw = 6
    vmin = -580
    vmax = 4290
    
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=vmin, vmax=vmax)

    ax3 = ax2.twinx()
    ax2.plot(FF['date'][:304:2],FF['WS'].rolling(window=2, center=True).mean()[:304:2],'-',c=c[1], zorder=10, label='wind speed')
    ax2.plot(FF['date'][314::60],FF['WS'].rolling(window=60, center=True).mean()[314::60],'-',c=c[1], zorder=10)
    ax3.plot((),(),'-',c=c[1], label='wind speed')
    ax3.plot(NB['date'],NB['cc'],'o',c=c[9], zorder=10, label='cloud cover')
    
    for d,t,i in zip(ttFF.columns[0::2],ttFF.columns[1::2],range(len(ttFF.columns[0::2]))):
        if i in [2,3,7]:
            pass
        else:
            ls = '-'
            if int(d[-1])%2 == 0:
                ls = ':'
            color = cmap(norm(ttFF_hordist[i]))
            pl = ax.plot(ttFF[d],ttFF[t],#.rolling(window=50, center=True).mean(),
                        #ls=ls,
                        color=color,#c=np.ones(np.shape(ttFF[d]))*(ttFF_hordist[i]),#(np.max(ttFF_elevation)-ttFF_elevation[i])/np.max(ttFF_elevation)),#cm.cool((8-i)/8),
                        #s=5,
                        #vmin=vmin,vmax=vmax,
                        #lw=5-(100-ttFF_waterdist[int(d[-1])-1])/50,
                        #label=f'{d[5:]} ({ttFF_height[int(d[-1])-1]}) [{ttFF_waterdist[int(d[-1])-1]}]', 
                        zorder=100)
    #ax.scatter(ttFF['date_FF1'],ttFF['t_FF1']-ttFF['t_FF9'],#.rolling(window=50, center=True).mean(),
    #                    #ls=ls,
    #                    c='r',#np.ones(np.shape(ttFF[d]))*(ttFF_elevation[i]),#(np.max(ttFF_elevation)-ttFF_elevation[i])/np.max(ttFF_elevation)),#cm.cool((8-i)/8),
    #                    s=100,
    #                    vmin=vmin,vmax=vmax,
    #                    #lw=5-(100-ttFF_waterdist[int(d[-1])-1])/50,
    #                    #label=f'{d[5:]} ({ttFF_height[int(d[-1])-1]}) [{ttFF_waterdist[int(d[-1])-1]}]', 
    #                    zorder=-10)
    #ttFF['date_FF9'] #ttFF['t_FF1']
    
    #ax3 = ax.twinx()
    #ax.get_shared_y_axes().join(ax, ax3)
    #for d,t,i in zip(ttNB.columns[-2::-2],ttNB.columns[-1::-2],range(len(ttNB.columns[0::2]))):
    #    if (ttNB.columns[-1::-2][i][2:4] in ['N1','N2','N6','N7']):
    #        print (t)#i, ttNB.columns[1::2][i][2:4])
    #        if t == 't_N7_':
    #            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.99),ls='--',label='N1', zorder=-10)
    #        elif t == 't_N6_':
    #            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.67),ls='--',label='N2', zorder=-10)
    #        elif t == 't_N2_':
    #            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.33),ls='--',label='N3', zorder=-10)
    #        elif t == 't_N1_':
    #            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(0),ls='--',label='N4', zorder=-10)
    
    #ax.plot(SM['date'],SM['T'],  '-',c=c[3],lw=lw, label='plateau',zorder=10)
    ax.scatter(NB['date'],NB['t_u'], label='glacier',
               marker='^',
               s=100,
               #zorder=10,
               vmin=vmin,vmax=vmax,
               ec='w',
               c=np.ones(np.shape(NB['date']))*(-583), zorder=1000)#520)) #inaccurate
    #ax.scatter(ttNB['date_N5_'], ttNB['t_N5_'], label='glacier tt',
    #           marker='*',
    #           s=100,
    #           #zorder=10,
    #           vmin=vmin,vmax=vmax,
    #           ec='w',
    #           c=np.ones(np.shape(ttNB['date_N5_']))*(-583))#520)) #inaccurate
    #ax.scatter(ttNB['date_N4_'], ttNB['t_N4_'], label='glacier tt',
    #           marker='s',
    #           s=100,
    #           #zorder=10,
    #           vmin=vmin,vmax=vmax,
    #           ec='w',
    #           c=np.ones(np.shape(ttNB['date_N5_']))*(-583))#520)) #inaccurate
    #ax.scatter(humilog_upper_hourly['date'],humilog_upper_hourly['T'],label='tongue (upper)',  
    #           marker='s',
    #           s=20,
    #           vmin=vmin,vmax=vmax,
    #           c=np.ones(np.shape(humilog_upper_hourly['date']))*(-238))#480))
    #ax.scatter(humilog_lower_hourly['date'],humilog_lower_hourly['T'],label='tongue (lower)',#.rolling(10).mean(),  
    #           marker='s',
    #           s=20,
    #           #zorder=100,
    #           vmin=vmin,vmax=vmax,
    #           c=np.ones(np.shape(humilog_lower_hourly['date']))*(-61))#426))
    ax.scatter(FF['date'][:304:2],FF['T'][:304:2],label='inlet',  #FF['date'][:304:2], FF['date'][314::60]
               marker='^',
               s=100,
               vmin=vmin,vmax=vmax,
               ec='w',
               c=np.ones(np.shape(FF['date'][:304:2]))*(829), zorder=1000)#308))
    ax.scatter(FF['date'][314::60],FF['T'][314::60],
               marker='^',
               s=100,
               vmin=vmin,vmax=vmax,
               ec='w',
               c=np.ones(np.shape(FF['date'][314::60]))*(829), zorder=1000)
    ax.scatter(NV['date'],NV['t'],label='outlet',  
               marker='^',
               s=100,
               vmin=vmin,vmax=vmax,
               ec='w',
               c=np.ones(np.shape(NV['date']))*(2700), zorder=1000)#285)) #tinytag elevation, not sure about station
    #ax.scatter(MG['date'],MG['T'],  
    #           marker='^',
    #           s=100,
    #           vmin=vmin,vmax=vmax,
    #           ec='w',
    #           c=np.ones(np.shape(MG['date']))*(4710))#305))
    #ax.plot(BH['date'],BH['t'],  '-',c=c[5],lw=lw, label='valley2',zorder=10)###########
    
    #dates = np.load(f'wrf_profiles-and-timeseries/surface_temp/dates-for-T2.npy')
    #markers = ['o','*']
    #for fl,tl in enumerate(['warm','cold']):
    #    ax.scatter(dates[:,0], np.load(f'wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_forefield.npy')[:,0], 
    #               vmin=vmin,vmax=vmax, c=np.ones(np.shape(dates[:,0]))*(720), marker=markers[fl], s=300, zorder=10000)
    #    ax.scatter(dates[:,0], np.load(f'wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_inlet.npy')[:,0], 
    #               vmin=vmin,vmax=vmax, c=np.ones(np.shape(dates[:,0]))*(829), marker=markers[fl], s=300, zorder=10000)
    #    ax.scatter(dates[:,0], np.load(f'wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_outlet.npy')[:,0], 
    #               vmin=vmin,vmax=vmax, c=np.ones(np.shape(dates[:,0]))*(2700), marker=markers[fl], s=300, zorder=10000)
    #    ax.scatter(dates[:,0], np.load(f'wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_valley-TT.npy')[:,0], 
    #               vmin=vmin,vmax=vmax, c=np.ones(np.shape(dates[:,0]))*(4290), marker=markers[fl], s=300, zorder=10000)
    
    #ax.axvline(np.datetime64('2023-09-13T08:00:00'),c=c[9],ls='--') # swapped hygroclip sensor
    ax.set_ylabel('temperature (\u00b0C)')#, c=c[0])
    ax.tick_params(axis='y')#, colors=c[0])
    
    ax2.axvspan(xmin=datetime(2023, 9, 12, 12, 0, 0), xmax=datetime(2023, 9, 14, 15, 0, 0), ymin=0, ymax=1, facecolor='grey', alpha=0.15)
    ax.axvspan(xmin=datetime(2023, 9, 12, 12, 0, 0), xmax=datetime(2023, 9, 14, 15, 0, 0), ymin=0, ymax=1, facecolor='grey', alpha=0.15)
    ax.set_xlim(datetime(2023,9,12,0,0,0),datetime(2023,9,22,0,0,0))
    ax2.set_xlim(datetime(2023,9,12,0,0,0),datetime(2023,9,22,0,0,0))
    
    #ax.xaxis.set_major_locator(mdates.DayLocator())
    # Specify the date format you want e.g. '2021-01-01' 
    date_format = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    
    ax.set_xlabel('local time')
    ax.set_ylim(6,13)
    ax.set_yticks(np.arange(0,16))
    fig.autofmt_xdate(rotation=45)
    
    ax2.set_ylim(0,7.5)
    ax2.set_yticks(np.arange(0,7.6,1.5))
    ax3.set_ylim(0,1)
    ax3.set_yticks(np.arange(0,1.01,.2))#[0,.25,.5,.75,1])
    ax3.set_yticklabels(np.arange(0,101,20))#[0,25,50,75,100])
    ax2.set_ylabel('wind speed (m s$^{-1}$)')
    ax3.set_ylabel('cloud cover (%)', rotation=270, labelpad=25)
    
    legend_1 = ax.legend(loc='best')
    ax3.legend(loc=7, bbox_to_anchor=(.99,0.4))
    
    ax2.grid()
    ax.grid()
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])#vmin,vmax])  # Needs to set an array for scalar mappable
    cb = plt.colorbar(sm, ax=ax, location='bottom',aspect=60)#,shrink=.5)
    cb.set_label('distance from glacier front (m down-valley)')#elevation (m a.s.l.)')
    plt.tight_layout()
    plt.savefig('plots/temp.pdf', format='pdf', dpi=80)
    
    plt.show()


# aws ----------------------------------------------------------------

#fig,ax = plt.subplots(figsize=(20,25),dpi=300)
#plt.rcParams.update({'font.size': 22})
#
#lw = 2#6
#
##ax.plot(SB['date'],SB['T'],  '-',c='k',lw=lw, label='mountain',zorder=10)
##ax.plot(SM['date'],SM['T'],  '-',c=c[3],lw=lw, label='plateau',zorder=10)
#ax.plot(NB['date'],NB['t_u'],'-',c=c[9],lw=lw, label='glacier',zorder=10)
#ax.plot(humilog_upper['date'],humilog_upper['T'],  '-',c=c[8],  lw=lw, label='tongue$_{upper}$',zorder=10)
#ax.plot(humilog_lower['date'],humilog_lower['T'],  '-',c='grey',lw=lw, label='tongue$_{lower}$',zorder=10)
#ax.plot(FF['date'],FF['T'],  '-',c=c[1],lw=lw, label='inlet',zorder=10)
#ax.plot(NV['date'],NV['t'],  '-',c=c[6],lw=lw, label='outlet',zorder=10)
#ax.plot(MG['date'],MG['T'],  '-',c=c[2],lw=lw, label='valley1',zorder=10)
#ax.plot(BH['date'],BH['t'],  '-',c=c[5],lw=lw, label='valley2',zorder=10)###########3
#
#ax2 = ax.twinx()
#ax2.sharey(ax)#get_shared_y_axes().join(ax, ax2)
##for d,t,i in zip(ttFF.columns[0::2],ttFF.columns[1::2],range(len(ttFF.columns[0::2]))):
##    ax2.plot(ttFF[d],ttFF[t],#.rolling(window=50, center=True).mean(),
##             c=cm.cool((8-i)/8),label=d[5:], zorder=-10)
#ax3 = ax.twinx()
#ax3.sharey(ax)#get_shared_y_axes().join(ax, ax3)
##for d,t,i in zip(ttNB.columns[-2::-2],ttNB.columns[-1::-2],range(len(ttNB.columns[0::2]))):
##    if (ttNB.columns[-1::-2][i][2:4] in ['N1','N2','N6','N7']):
##        print (t)#i, ttNB.columns[1::2][i][2:4])
##        if t == 't_N7_':
##            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.99),ls='--',label='N1', zorder=-10)
##        elif t == 't_N6_':
##            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.67),ls='--',label='N2', zorder=-10)
##        elif t == 't_N2_':
##            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.33),ls='--',label='N3', zorder=-10)
##        elif t == 't_N1_':
##            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(0),ls='--',label='N4', zorder=-10)
#
#ax.axvline(np.datetime64('2023-09-13T08:00:00'),c=c[9],ls='--') # swapped hygroclip sensor
#ax.axvline(np.datetime64('2023-09-12T16:00:00'),c=c[9],ls='--',lw=4) # swapped radiometer
#ax.axvline(np.datetime64('2023-09-12T17:00:00'),c=c[9],ls='--',lw=4) # swapped radiometer
#
#ax.set_ylabel('temperarure (\u00b0C)')#, c=c[0])
#ax.tick_params(axis='y')#, colors=c[0])
#
#ax.axvspan(xmin=IOP_start, xmax=IOP_end, ymin=0, ymax=1, facecolor='grey', alpha=0.1)
#ax.set_xlim(datetime(2023, 9, 13, 0, 0, 0), datetime(2023, 9, 15, 0, 0, 0))#end)
#ax.set_xlabel('local time')
#ax.set_ylim(2,15)
#fig.autofmt_xdate(rotation=45)
#
#legend_1 = ax.legend(loc='best')
#legend_1.remove()
#ax2.add_artist(legend_1)
#
#ax.grid()
#plt.savefig('plots/temp.png', format='png')
#
#plt.show()

# forefield ----------------------------------------------------------

#fig,ax = plt.subplots(figsize=(20,17),dpi=300)
#plt.rcParams.update({'font.size': 22})
#
#lw = 6
#
##ax.plot(SM['date'],SM['T'],  '-',c=c[3],lw=lw, label='plateau',zorder=10)
##ax.plot(NB['date'],NB['t_u'],'-',c=c[9],lw=lw, label='glacier',zorder=10)
##ax.plot(humilog['date'],humilog['T'],  '-',c=c[8],lw=lw, label='tongue',zorder=10)
##ax.plot(FF['date'],FF['T'],  '-',c=c[1],lw=lw, label='inlet',zorder=10)
##ax.plot(NV['date'],NV['t'],  '-',c=c[6],lw=lw, label='outlet',zorder=10)
##ax.plot(MG['date'],MG['T'],  '-',c=c[2],lw=lw, label='valley1',zorder=10)
##ax.plot(BH['date'],BH['t'],  '-',c=c[5],lw=lw, label='valley2',zorder=10)
#
#ttFF_waterdist = [105,199,29,5,20,33,162,27,80] # NB! only horisontal, doesn't account for elevation (HOBO: 77)
#ttFF_height = [190,270,205,197,190,205,200,170,170]
#
#ax2 = ax.twinx()
#ax2.sharey(ax)#get_shared_y_axes().join(ax, ax2)
#for d,t,i in zip(ttFF.columns[0::2],ttFF.columns[1::2],range(len(ttFF.columns[0::2]))):
#    print (5+(100-ttFF_waterdist[int(d[-1])-1])/33)
#    ls = '-'
#    if int(d[-1])%2 == 0:
#        ls = ':'
#    ax2.plot(ttFF[d],ttFF[t],#.rolling(window=50, center=True).mean(),
#             ls=ls,c=cm.cool((8-i)/8),
#             #lw=5-(100-ttFF_waterdist[int(d[-1])-1])/50,
#             label=f'{d[5:]} ({ttFF_height[int(d[-1])-1]}) [{ttFF_waterdist[int(d[-1])-1]}]', 
#             zorder=-10)
#ax3 = ax.twinx()
#ax3.sharey(ax)##get_shared_y_axes().join(ax, ax3)
##for d,t,i in zip(ttNB.columns[-2::-2],ttNB.columns[-1::-2],range(len(ttNB.columns[0::2]))):
##    if (ttNB.columns[-1::-2][i][2:4] in ['N1','N2','N6','N7']):
##        print (t)#i, ttNB.columns[1::2][i][2:4])
##        if t == 't_N7_':
##            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.99),ls='--',label='N1', zorder=-10)
##        elif t == 't_N6_':
##            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.67),ls='--',label='N2', zorder=-10)
##        elif t == 't_N2_':
##            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.33),ls='--',label='N3', zorder=-10)
##        elif t == 't_N1_':
##            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(0),ls='--',label='N4', zorder=-10)
#
##ax.axvline(np.datetime64('2023-09-13T08:00:00'),c=c[9],ls='--') # swapped hygroclip sensor
#ax.set_ylabel('temperature (\u00b0C)')#, c=c[0])
#ax.tick_params(axis='y')#, colors=c[0])
#
#ax.axvspan(xmin=IOP_start, xmax=IOP_end, ymin=0, ymax=1, facecolor='grey', alpha=0.1)
#ax.set_xlim(start,end)
#ax.set_xlim(datetime(2023,9,11,0,0,0),datetime(2023,9,16,0,0,0))
#ax.set_xlabel('local time')
#ax.set_ylim(0,17)
#ax2.set_yticks([])
#ax3.set_yticks([])
#fig.autofmt_xdate(rotation=45)
#ax2.legend(ncol=3,loc=1)
#
#ax.grid()
#plt.savefig('plots/plots_campaign/temp_forefield.png', format='png')
#
#plt.show()

# glacier ------------------------------------------------------------

#fig,ax = plt.subplots(figsize=(20,17),dpi=300)
#plt.rcParams.update({'font.size': 22})
#
#lw = 6
#
#ax.plot(NB['date']+pd.Timedelta(minutes=0),NB['t_u'],'-',c=c[9],lw=lw, label='AWS',zorder=10) # 30 min offset for centering because hourly values are based on previous hour
#
#ax2 = ax.twinx()
#ax.get_shared_y_axes().join(ax, ax2)
#ax3 = ax.twinx()
#ax.get_shared_y_axes().join(ax, ax3)
#for d,t,i in zip(ttNB.columns[-2::-2],ttNB.columns[-1::-2],range(len(ttNB.columns[0::2]))):
#    if (ttNB.columns[-1::-2][i][2:4] in ['N1','N2','N6','N7']):
#        print (t)#i, ttNB.columns[1::2][i][2:4])
#        if t == 't_N7_':
#            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.99),ls='-',label='NB1', zorder=-10)
#        elif t == 't_N6_':
#            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.67),ls='-',label='NB2', zorder=-10)
#        elif t == 't_N2_':
#            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(.33),ls='-',label='NB3', zorder=-10)
#        elif t == 't_N1_':
#            ax3.plot(ttNB[d],ttNB[t],c=cm.cool(0),ls='-',label='NB4', zorder=-10)
#
#ax.axvline(np.datetime64('2023-09-12T17:36:00'),c=cm.cool(.99),ls='--') # swapped N1
#ax.axvline(np.datetime64('2023-09-12T16:37:00'),c=cm.cool(.67),ls='--') # swapped N2
#ax.axvline(np.datetime64('2023-09-12T14:43:00'),c=cm.cool(.33),ls='--') # swapped N3
#ax.axvline(np.datetime64('2023-09-12T14:37:00'),c=cm.cool(0),ls='--') # swapped N4
#ax.axvline(np.datetime64('2023-09-12T16:00:00'),c=c[9],ls='--',lw=4) # swapped radiometer
#ax.axvline(np.datetime64('2023-09-12T17:00:00'),c=c[9],ls='--',lw=4) # swapped radiometer
#ax.axvline(np.datetime64('2023-09-13T08:00:00'),c=c[9],ls='--') # swapped hygroclip sensor
#ax.set_ylabel('temperarure (\u00b0C)')#, c=c[0])
#ax.tick_params(axis='y')#, colors=c[0])
#
#ax.axvspan(xmin=IOP_start, xmax=IOP_end, ymin=0, ymax=1, facecolor='grey', alpha=0.1)
#ax.set_xlim(start,end)
#ax.set_xlabel('local time')
#ax.set_ylim(3,12)
#ax2.set_yticks([])
#ax3.set_yticks([])
#fig.autofmt_xdate(rotation=45)
##ax2.legend(ncol=3,loc=2)
#ax3.legend(ncol=2,loc=1)
#
#legend_1 = ax.legend(loc=2)
#legend_1.remove()
#ax2.add_artist(legend_1)
##ax3.add_artist(legend_1)
#
#ax.grid(True)
#fig.tight_layout()
#plt.savefig('plots/plots_campaign/temp_transect.png', format='png')
#
#plt.show()

# --------------------------------------------------------------------
# plot humidity
# --------------------------------------------------------------------

def plotHumidity():
        
    plt.rcParams.update({'font.size': 22})
    fig,ax = plt.subplots(figsize=(20,8),dpi=300)
    
    #ax.plot(NB['date']+pd.Timedelta(minutes=30),NB['rh_u'],'-',c=c[9], label='glacier')
    ax.plot(humilog_upper['date'],humilog_upper['RH'], '-', lw=lw/2, c=c[8],   label='tongue$_{upper}$')
    ax.plot(humilog_lower['date'],humilog_lower['RH'], '-', lw=lw/2, c='grey', label='tongue$_{lower}$')
    ax.plot(FF['date'],FF['RH'], '-', lw=lw/2, c=c[1], label='inlet')
    
    ax.axvspan(xmin=IOP_start, xmax=IOP_end, ymin=0, ymax=1, facecolor='grey', alpha=0.1)
    #ax.axvline(np.datetime64('2023-09-12T16:00:00'),c=c[9],ls='--') # swapped radiometer
    #ax.axvline(np.datetime64('2023-09-12T17:00:00'),c=c[9],ls='--') # swapped radiometer
    #ax.axvline(np.datetime64('2023-09-13T08:00:00'),c=c[9],ls='--') # swapped hygroclip sensor
    ax.set_ylabel('relative humidity (%)')#, c=c[0])
    ax.set_ylim(27,100)
    ax.tick_params(axis='y')#, colors=c[0])
    
    ax2 = ax.twinx()
    #ax.plot(NB['date'],NB['t_u'],'-',c=c[9],lw=lw, label='glacier',zorder=10)
    ax2.plot(humilog_upper['date'],humilog_upper['T'], ':', c=c[8], zorder=10)
    ax2.plot(humilog_lower['date'],humilog_lower['T'], ':', c='grey', zorder=10)
    ax2.plot(FF['date'],FF['T'], ':', c=c[1], zorder=10)
    #ax.plot(NV['date'],NV['t'],'-',c=c[6],lw=lw, label='outlet',zorder=10)
    #ax.plot(MG['date'],MG['T'],'-',c=c[2],lw=lw, label='valley1',zorder=10)
    #ax.plot(BH['date'],BH['t'],'-',c=c[5],lw=lw, label='valley2',zorder=10)
    
    ax2.plot((),(),  '-', lw=lw/2, c='k', label='relative humidity',zorder=10)
    ax2.plot((),(),  ':', c='k', label='temperature',zorder=10)
    ax2.set_ylabel('temperature (\u00b0C)', rotation=270, labelpad=25)
    ax2.set_ylim(0,12)
    ax2.legend(loc=3)
    
    ax.set_xlim(start,end)
    ax.set_xlim(datetime(2023,9,13,0,0,0),datetime(2023,9,15,0,0,0))
    ax.set_xlabel('local time')
    fig.autofmt_xdate(rotation=45)
    ax.legend(loc=4)
    plt.grid()
    plt.savefig('plots/plots_campaign/rh.png', format='png')
    #plt.savefig('plots/plots_campaign/rh_ext.png', format='png')
    
    plt.show()


# --------------------------------------------------------------------
# plot pressure
# --------------------------------------------------------------------

def plotPressure():

    plt.rcParams.update({'font.size': 22})
    fig,ax = plt.subplots(figsize=(20,8),dpi=300)
    
    ax.plot(NB['date'],NB['p_u'],'-',c=c[9], label='glacier')
    
    ax.axvspan(xmin=IOP_start, xmax=IOP_end, ymin=0, ymax=1, facecolor='grey', alpha=0.1)
    #ax.axvline(np.datetime64('2023-09-12T16:00:00'),c=c[9],ls='--') # swapped radiometer
    #ax.axvline(np.datetime64('2023-09-12T17:00:00'),c=c[9],ls='--') # swapped radiometer
    ax.axvline(np.datetime64('2023-09-13T08:00:00'),c=c[9],ls='--') # swapped hygroclip sensor
    ax.set_ylabel('pressure (hPa)')#, c=c[0])
    ax.tick_params(axis='y')#, colors=c[0])
    
    ax.set_xlim(start,end)
    #ax.set_xlim(datetime(2023,9,12,12,0,0),datetime(2023,10,18,12,0,0))
    ax.set_xlabel('local time')
    fig.autofmt_xdate(rotation=45)
    plt.legend(loc=8)
    plt.grid()
    #plt.savefig('plots/plots_campaign/rh.png', format='png')
    #plt.savefig('plots/plots_campaign/rh_ext2.png', format='png')
    
    plt.show()

# --------------------------------------------------------------------
# plot radiation
# --------------------------------------------------------------------

def plotRadiation():
    
    plt.rcParams.update({'font.size': 22})
    fig,ax = plt.subplots(figsize=(20,8),dpi=300)
    
    #ax.plot(SM['date'],SM['SW_in'],  '-',c=c[3], label='plateau')
    ax.plot(NB['date']+pd.Timedelta(minutes=30),NB['dsr_cor'],'-',c=c[9], label='glacier')
    ax.plot(FF['date'],FF['SW_in'],  '-',c=c[1], label='inlet')
    ax.plot(NB['date'],NB['usr_cor'],':',c=c[9])
    ax.plot(FF['date'],FF['SW_out'], ':',c=c[1])
    ax.plot(NB['date'],NB['dlr'],'-',c=c[9])
    ax.plot(NB['date'],NB['ulr'],':',c=c[9])
    ax2 = ax.twinx()
    ax2.scatter(NB['date'],NB['cc'],marker='o',c=c[9])
    ax.plot((),(),'-',c='k',label='in')
    ax.plot((),(),':',c='k',label='out')
    
    ax.axvspan(xmin=IOP_start, xmax=IOP_end, ymin=0, ymax=1, facecolor='grey', alpha=0.1)
    ax.axvline(np.datetime64('2023-09-14T13:45:00'),c=c[1],ls='--') # rotated radiometer on hobo station
    ax.set_ylabel('shortwave radiation (W/m${^2}$)')
    
    ax.set_xlim(start,end)
    ax.set_xlim(datetime(2023,9,11,0,0,0),datetime(2023,9,19,0,0,0))
    ax.set_xlabel('local time')
    ax.set_ylim(0,750)
    fig.autofmt_xdate(rotation=45)
    ax.legend()
    plt.grid()
    plt.savefig('plots/plots_campaign/rad.png', format='png')
    
    plt.show()


# --------------------------------------------------------------------
# plot precipitation
# --------------------------------------------------------------------

def plotPrecipitation():

    #start = datetime(2023,  9, 17, 0, 0, 0)
    #end   = datetime(2023, 10, 17, 0, 0, 0)
    
    plt.rcParams.update({'font.size': 22})
    fig,ax = plt.subplots(figsize=(20,8),dpi=300)
    offset = pd.Timedelta(hours=4,minutes=11)
    w=.1#.01
    
    ax.bar(FF['date']-offset,FF['PREC_day'],width=w,color=c[1], label='inlet')
    ax.bar(MG['date']       ,MG['PREC_day'],width=w,color=c[2], label='valley1')
    ax.bar(BH['date']+offset,BH['prec_day'],width=w,color=c[5], label='valley2')
    ax.axvline(np.datetime64('2023-09-15T17:00:00'),c=c[1],ls='--') # activated rain gauge
    ax.set_ylabel('precipitation (mm/h)')
    
    ax.set_xlim(start,end)
    ax.set_xlim(datetime(2023,  9, 11, 0, 0, 0),datetime(2023, 10, 17, 0, 0, 0))
    ax.set_xlabel('local time')
    #ax.set_ylim(0,2)
    fig.autofmt_xdate(rotation=45)
    plt.legend(loc=9)
    plt.grid()
    #plt.savefig('plots/plots_extended-period/precip.png', format='png')
    
    plt.show()


# --------------------------------------------------------------------
# interpolate and plot UAV data
# --------------------------------------------------------------------

# load already stored intalts
def load_intalts(t):
    with open(f'data/interpolation/intalts/intalts_{t[:5]}.txt', "r") as file:
        intalts = file.read()
        intalts = ast.literal_eval(intalts)
    return intalts
def load_intaltsext(t):
    with open(f'data/interpolation/intalts/intaltsext_{t[:5]}.txt', "r") as file:
        intalts_extended = file.read()
        intalts_extended = ast.literal_eval(intalts_extended)
    return intalts_extended

def get_elevation(lon, lat, dataset):
    # Transform coordinates to the dataset's CRS
    transformer = Transformer.from_crs("epsg:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)

    # Get row and column in the image for the given coordinate
    row, col = rowcol(dataset.transform, x, y)

    # Check if the row and column are within the bounds of the image
    if (0 <= row < dataset.height) and (0 <= col < dataset.width):
        # Read the dataset's values at the discovered row and column
        value = dataset.read(1)[row, col]
        return value
    else:
        # Return NaN if the coordinate is outside the raster
        return np.nan
    return value

def get_intalts(intpts, path, dem_hd_path, update_intalts_from_DEM): #=dem_uav_path):
    if update_intalts_from_DEM == True:
        with rasterio.open(path) as dataset:
            elevations_uav = [get_elevation(lon, lat, dataset) for lat, lon in intpts]
        with rasterio.open(dem_hd_path) as dataset:
            elevations_hd = [get_elevation(lon, lat, dataset) for lat, lon in intpts]
        
    elevations_uav = [e if e > 0 else np.nan for e in elevations_uav] #375
    elevations_hd = [e if e > 0 else np.nan for e in elevations_hd] #375 
    return elevations_uav, elevations_hd #intalts


def interpolate_array(array, n, x, arraytype='normal'):
    original_length = len(array)
    original_indices = np.linspace(0, original_length-1, original_length)
    new_length = (n-1)*x + 1
    new_indices = np.linspace(0, original_length-1, new_length)
    if arraytype == 'coord':
        interpolated_lat = np.interp(new_indices, original_indices, [coord[0] for coord in array])
        interpolated_lon = np.interp(new_indices, original_indices, [coord[1] for coord in array])
        interpolated_array = list(zip(interpolated_lat, interpolated_lon))
    else:
        interpolated_array = np.interp(new_indices, original_indices, array)
    return interpolated_array

def get_intaltsext(intpts, path, dem_hd_path): #=dem_uav_path):
    # create extended intalts for final plot
    intpts_extended = interpolate_array(intpts, len(intpts), 10, 'coord')
#    print ('diff in intpts and intptsext: ', np.array(intpts_extended[::10])-np.array(intpts))
    with rasterio.open(path) as dataset:
        elevations_uav = [get_elevation(lon, lat, dataset) for lat, lon in intpts_extended]
    with rasterio.open(dem_hd_path) as dataset:
        elevations_hd = [get_elevation(lon, lat, dataset) for lat, lon in intpts_extended]
        
    elevations_uav = [e if e > 0 else np.nan for e in elevations_uav]
    elevations_hd = [e if e > 0 else np.nan for e in elevations_hd] 
    return elevations_uav, elevations_hd


def interpolate_coordinates(coordinates, n, x):
    original_length = len(coordinates)
    original_indices = np.linspace(0, original_length-1, original_length)
    new_length = n*x + 1
    new_indices = np.linspace(0, original_length-1, new_length)

    interpolated_lat = np.interp(new_indices, original_indices, [coord[0] for coord in coordinates])
    interpolated_lon = np.interp(new_indices, original_indices, [coord[1] for coord in coordinates])

    interpolated_coordinates = list(zip(interpolated_lat, interpolated_lon))
    return interpolated_coordinates


def add_obs(interpolated_dist_z, target_dists, target_zs, obsloc, xind_inlet, var='t', ndint=False):
    if obsloc == 'outlet':
        xind = np.where(target_dists >= xind_outlet)[1][0]
        zind = np.where(target_zs == round(outlet_aws[2]))[0][0]
        if ndint == False:
            zind = 0
        tobs = NV.loc[NV['date']>=date,'t'].iloc[0]
    elif obsloc == 'inlet':
        xind = np.where(target_dists >= xind_inlet)[1][0]
        zind = np.where(target_zs == round(inlet_aws[2]))[0][0]
        if ndint == False:
            zind = 0
        if var == 't':
            tobs = FF.loc[FF['date']>=date,'T'].iloc[0]
        elif var == 'rh':
            tobs = FF.loc[FF['date']>=date,'RH'].iloc[0]
    print (xind,zind)
    print (f'old value: {interpolated_dist_z[zind,xind]}')
    interpolated_dist_z[zind,xind] = tobs
    print (f'new value: {interpolated_dist_z[zind,xind]}')
    
    if ndint == True:
        # update interpolated field with new point
        non_nan_indices = np.where(~np.isnan(interpolated_dist_z))
        points = np.column_stack(non_nan_indices)
        values = interpolated_dist_z[non_nan_indices]
        interpolator = LinearNDInterpolator(points, values)
        all_indices = np.indices(interpolated_dist_z.shape)
        interpolated_dist_z = interpolator(all_indices[0],all_indices[1])
    return interpolated_dist_z
    

# part 1 of interpolation procedure ----------------------------------
    
glacier_aws =  (7.197794684331172, 61.686051540061946, 550)
upper_tongue = (7.198646470752097, 61.683231702332485, 480)
lower_tongue = (7.198529491371879, 61.6814934541163,   462)
# front @ 493-495 m
inlet_aws =    (7.211611010507808, 61.675952354678884, 277)
outlet_aws =   (7.2415509,         61.6672661,         276)

pt_gla = Point(glacier_aws[1], glacier_aws[0]) # glacier
pt_tup = Point(upper_tongue[1], upper_tongue[0]) # upper tongue
pt_tlo = Point(lower_tongue[1], lower_tongue[0]) # lower tongue
pt_fro = Point(61.6805, 7.1993) # front
pt_inl = Point(inlet_aws[1], inlet_aws[0]) # inlet
pt_out = Point(outlet_aws[1], outlet_aws[0]) # outlet

# find elevation of points from updated DEM
#dem_uav_path = 'data/DEM/Nigardsbreen140923_10cm_DSM.tif'
dem_uav_path = 'data/DEM/Nigardsbreen_Wingtra_DSM_aligned.tif'
dem_hd_path = 'data/DEM/Jostedalsbreen_10m.tif'
update_intalts_from_DEM = True

def interpolateUAV_p1(ind1=0, ind2=2, day='13'):
    
    path = 'data/UAV/iMet/'
    extension = 'csv'
    
    fns = glob.glob(path+'*.{}'.format(extension))
    imets = {}

    profiles = ['5','5-US','7','7-US','10','10-US','13','13-US','16']
    profiles = profiles[ind1:ind2]#['11']#
    
    outlet = False
    inlet = True#False
    
    for ph in profiles:
        for fn in fns:
            #print (fn[len(path)+6:len(path)+8],fn[len(path)+25:len(path)+33])
            if (fn[len(path)+6:len(path)+8] == day):# and (fn[-14:] != 'paraglider.csv'):
                imet = pd.read_csv(fn, sep=',', dtype='string', 
                                   usecols=['XQ-iMet-XQ Pressure', 'XQ-iMet-XQ Air Temperature', 'XQ-iMet-XQ Humidity', \
                                            'XQ-iMet-XQ Date', 'XQ-iMet-XQ Time','XQ-iMet-XQ Longitude', \
                                            'XQ-iMet-XQ Latitude', 'XQ-iMet-XQ Altitude', 'Profile hour'])
                if imet.empty:
                    print(f'DataFrame is empty!')
                else:
                    imet = imet.rename(columns={'XQ-iMet-XQ Pressure': 'p', 'XQ-iMet-XQ Air Temperature': 't', \
                                                'XQ-iMet-XQ Humidity': 'rh', 'XQ-iMet-XQ Date': 'Date', \
                                                'XQ-iMet-XQ Time': 'time', 'XQ-iMet-XQ Longitude': 'lon', \
                                                'XQ-iMet-XQ Latitude': 'lat', 'XQ-iMet-XQ Altitude': 'z', \
                                                'Profile hour': 'hour'})
                    imet = (imet.astype({col: float for col in imet.columns[:3]}))
                    imet = (imet.astype({col: float for col in imet.columns[5:8]}))
                    imet['pt'] = (imet['t']+273.15)*(1013/imet['p'])**(0.286)
                    if fn[-14:] == 'paraglider.csv' or (fn[len(path)+6:len(path)+8] == '12' and fn[-9:] == 'front.csv'):
                        imet['date'] = pd.to_datetime(imet['Date']+' '+imet['time'], format='%d/%m/%Y %H:%M:%S')
                    else:
                        imet['date'] = pd.to_datetime(imet['Date']+' '+imet['time'], format='%Y/%m/%d %H:%M:%S')
                    del imet['Date'], imet['time']
    
                    if fn[len(path)+25:len(path)+28] != 'gla':
                        imet_tmp = imet.loc[(imet['hour'])==ph]
                        if imet_tmp.empty != True:
                            if fn[-14:] != 'paraglider.csv':
                                imets[fn[len(path)+6:len(path)+8]+'-'+ph+'-'+fn[len(path)+25:len(path)+28]] = imet_tmp.reset_index(drop=True)
                            else:
                                print ('using paraglider data')
                                imets[fn[len(path)+6:len(path)+8]+'-'+ph+'-'+fn[len(path)+25:len(path)+28]+'aglider'] = imet_tmp.reset_index(drop=True)
                    else:
                        imet_tmp = imet.loc[imet['hour']==ph+'-UP']
                        if imet_tmp.empty != True:
                            imets[fn[len(path)+6:len(path)+8]+'-'+ph+'-'+fn[len(path)+25:len(path)+28]+'-upper'] = imet_tmp.reset_index(drop=True)
                        imet_tmp = imet.loc[imet['hour']==ph+'-LOW']
                        if imet_tmp.empty != True:
                            imets[fn[len(path)+6:len(path)+8]+'-'+ph+'-'+fn[len(path)+25:len(path)+28]+'-lower'] = imet_tmp.reset_index(drop=True)
                        #imet_tmp = imet.loc[imet['hour']==ph+'-extra']
                        #if imet_tmp.empty != True:
                        #    imets[fn[len(path)+6:len(path)+8]+'-'+ph+'-'+fn[len(path)+25:len(path)+28]+'-extra'] = imet_tmp.reset_index(drop=True)
                    if imet_tmp.empty:
                        print (fn[len(path)+6:len(path)+8]+'-'+ph+'-'+fn[len(path)+25:len(path)+35])
                    elif fn[len(path)+25:len(path)+28] == 'out':
                        #print (fn, imet_tmp)
                        outlet = True
                    elif fn[len(path)+25:len(path)+28] == 'in':
                        inlet = True
    
    sorted_keys = sorted(imets, key=lambda x: imets[x].loc[0, 'lat'])
    imets = {key: imets[key] for key in sorted_keys}
        
    # read data
    im = plt.imread('data/map/2022-08-31-00_00_2022-08-31-23_59_Sentinel-2_L1C_True_color_Gjerde.png')
    
    fig,ax = plt.subplots()
    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)
    
    xmin=7.111416
    xmax=7.30299
    ymin=61.650282
    ymax=61.705824
    
    ax.imshow(im, extent=[xmin,xmax,ymin,ymax], aspect='auto')
    
    lons = []
    lats = []
    alts = []
    
    if outlet == False:
        print ('adding outlet')
        lons.append(outlet_aws[0])
        lats.append(outlet_aws[1])
        alts.append(outlet_aws[2])
    if inlet == False:
        print ('adding inlet')
        lons.append(inlet_aws[0])
        lats.append(inlet_aws[1])
        alts.append(inlet_aws[2])
    
    for t in imets:#profiles:#times:
        if imets[t].empty:
            print(f'DataFrame {t} is empty!') # shouldn't be necessary any longer
        else:
            imet_tmp = imets[t]
    
            # define points for interpolation line
            if t[-10:] != 'paraglider':
                lons.append(imet_tmp['lon'].values[0])
                lats.append(imet_tmp['lat'].values[0])
                alts.append(imet_tmp['z'].values[0])
            else:
                print ('adding paraglider')
                lons.append(imet_tmp['lon'].values.mean())
                lats.append(imet_tmp['lat'].values.mean())
                alts.append(280)#imet_tmp['z'].values[-1])
            ax.scatter(imet_tmp['lon'],imet_tmp['lat'],s=30,label=f'{t}')
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            plt.legend()
    
            
    intpts = []
    intline = {}

    intline[t[:4]] = pd.DataFrame(columns=['lons','lats','alts','pts'])
    intline[(t[:4])]['lons'] = lons
    intline[t[:4]]['lats'] = lats
    intline[t[:4]]['alts'] = alts
    intline[t[:4]] = intline[t[:4]].sort_values(by=['lats']).reset_index(drop=True) # shouldn't be necessary any longer
    for i in range(len(intline[t[:4]]['alts'].values)):
        intline[t[:4]]['pts'][i] = Point(intline[t[:4]]['lats'][i],intline[t[:4]]['lons'][i])
        if i == 0:
            intpts.extend(([intline[t[:4]]['pts'][0].coords[0]]))
        else:
            line = (LineString([intline[t[:4]]['pts'][i-1],intline[t[:4]]['pts'][i]]))
            for div in np.arange(0.1,1,0.1):
                intpts.extend(line.interpolate(div, normalized=True).coords[:])
            intpts.extend(([intline[t[:4]]['pts'][i].coords[0]]))
    
    for i in range(len(intpts)):
        ax.scatter(intpts[i][1],intpts[i][0],c='k',s=10)
        if (i%10) == 0:
            ax.scatter(intpts[i][1],intpts[i][0],c='k',s=20)
    ax.scatter(glacier_aws[0], glacier_aws[1], c='k', marker='x', s=60)
    ax.scatter(upper_tongue[0],upper_tongue[1], c='k', marker='x', s=60)
    ax.scatter(lower_tongue[0],lower_tongue[1], c='k', marker='x', s=60)
    ax.scatter(inlet_aws[0],inlet_aws[1], c='k', marker='x', s=60)
    ax.scatter(outlet_aws[0],outlet_aws[1], c='k', marker='x', s=60)
    
    # find index where balloon has drifted more than 0.002 in latitude direction:
    if t[:5] == '13-5-':
        max_height = RS4.loc[RS4_ind,'z']
        print (f'location of radiosonde below {max_height}')
        ax.scatter(RS4.loc[:RS4_ind,'lon'], RS4.loc[:RS4_ind,'lat'], c='y', marker='^', s=10)
    
    if t[:5] == '12-11':
        max_height = RS2.loc[RS2_ind,'z']
        print (f'location of radiosonde below {max_height}')
        ax.scatter(RS2.loc[:RS2_ind,'lon'], RS2.loc[:RS2_ind,'lat'], c='y', marker='^', s=10)
    
    #plt.savefig(f'plots/map_{t[:5]}.png')
    
    # find surface altitude of each coordinate on interpolation line
    try:
        intalts = load_intalts(t) #intalts# = intalts_stored[t[:5]]
    except:
        elevations_uav, elevations_hd = get_intalts(intpts, dem_uav_path, dem_hd_path, update_intalts_from_DEM) # dem_uav_path or dem_hd_path
        intalts_hd = elevations_hd
        intalts = [elevations_uav[i] if not np.isnan(elevations_uav[i]) else elevations_hd[i] for i in range(len(elevations_uav))]
        # save altitudes from DEM along interpolation line to file
        with open(f'data/interpolation/intalts/intalts_{t[:5]}.txt', 'w') as f:
            f.write(f"{intalts}\n")
        print ('created new intalts')
    else:
        print ('using already stored intalts')
    
    try:
        intalts_extended = load_intaltsext(t) #intalts# = intalts_stored[t[:5]]
    except:
        elevations_uav, elevations_hd = get_intaltsext(intpts, dem_uav_path, dem_hd_path)
        intalts_extended_hd = elevations_hd
        intalts_extended = [elevations_uav[i] if not np.isnan(elevations_uav[i]) else elevations_hd[i] for i in range(len(elevations_uav))]
        with open(f'data/interpolation/intalts/intaltsext_{t[:5]}.txt', 'w') as f:
            f.write(f"{intalts_extended}\n")
        print ('created new intalts_extended')
    else:
        print ('using already stored intalts_extended')
    
    # calculate distance along interpolation line
    intdist = [0]
    geod = Geod(ellps="WGS84")
    
    for i in range(1,len(intpts)):
        line_string = (LineString([intpts[i],intpts[i-1]]))
        intdist.extend([intdist[-1]+geod.geometry_length(line_string)])

    # check interpolation line
    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(intdist, intalts_extended[::10], marker='o', linestyle='-', c='blue')
    #plt.plot(intdist, intalts_extended_hd[::10], marker='o', linestyle='-', c='red')
    plt.plot(intdist, intalts, marker='o', linestyle='-', c='y')
    for i in range(0,len(intdist),10):
        plt.axvline(intdist[i])
    plt.axhline(375)
    plt.title('x, z')
    plt.xlabel('distance (m)')
    plt.ylabel('elevation (m a.s.l.)')
    plt.grid(True)
    plt.show()
    
    # adding outlet data to dateframe dictionary if missing
    fs = list(imets.keys())[0]
    day = fs[:2]
    hour = fs[3:4]
    if fs[3] == '1':
        hour = fs[3:5]
    date = datetime(2023,9,int(day),int(hour)+2,0,0)
    if len(intalts[::10])-len(imets) == 1:
        print ('let us add something')
        if outlet == False and inlet == True:
            print ('adding outlet observation')
            sorted_keys_new = np.append(f'{fs[:5]}-outlet', sorted_keys)#, list(imets.keys())
            imets[f'{fs[:5]}-outlet'] = pd.DataFrame(columns=imets[sorted_keys[0]].columns)
            imets[f'{fs[:5]}-outlet'].loc[0] = np.ones(len(imets[f'{fs[:5]}-outlet'].columns))*np.nan
            imets[f'{fs[:5]}-outlet'].loc[0,'t'] = NV.loc[NV['date']>=date,'t'].iloc[0]
            imets[f'{fs[:5]}-outlet'].loc[0,'lon'] = outlet_aws[0]
            imets[f'{fs[:5]}-outlet'].loc[0,'lat'] = outlet_aws[1]
            imets[f'{fs[:5]}-outlet'].loc[0,'z'] = outlet_aws[2]
            imets = ({k: imets[k] for k in sorted_keys_new})
            #print (imets[f'{fs[:5]}-outlet'])
        elif outlet == True and inlet == False:
            print ('adding inlet observation')
            sorted_keys.insert(1,f'{fs[:5]}-inlet') # preparing to add df to 2nd entry of dictionary
            imets[f'{fs[:5]}-inlet'] = pd.DataFrame(columns=imets[sorted_keys[0]].columns)
            imets[f'{fs[:5]}-inlet'].loc[0] = np.ones(len(imets[f'{fs[:5]}-inlet'].columns))*np.nan
            imets[f'{fs[:5]}-inlet'].loc[0,'t'] = FF.loc[FF['date']>=date,'T'].iloc[0]
            imets[f'{fs[:5]}-inlet'].loc[0,'lon'] = inlet_aws[0]
            imets[f'{fs[:5]}-inlet'].loc[0,'lat'] = inlet_aws[1]
            imets[f'{fs[:5]}-inlet'].loc[0,'z'] = inlet_aws[2]
            
            imets = ({k: imets[k] for k in sorted_keys})

    return t, imets, intdist, intpts, intalts, intalts_extended, date




# part 2 of interpolation procedure ----------------------------------

def interpolateUAV_p2(var='t', date=date, pt_contours=None):
    
    processed_dfs = []
    
    var = var#'t'
    print (var)
    
    #if var == 't':
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
        
    selected_columns = ['z',var]
    
    for df_tmp in imets:
        if imets[df_tmp].empty:
            print(f'DataFrame is empty!')
        else:
            # Select specific columns
            df_selected = imets[df_tmp][selected_columns]
    
            # for ascents:
            # Add a column to track the cumulative maximum of 'z' so far for each row    
            df_selected['max_z'] = df_selected['z'].copy() ######
            df_selected['max_z'] = df_selected['max_z'].cummax()
            
            if df_tmp != '13-13-paraglider':
                # Filter the DataFrame to keep only the rows where 'z' equals its cumulative maximum (except paraglider flight)
                df_selected = df_selected[df_selected['z'] == df_selected['max_z']].drop(columns='max_z')
            else:
                df_selected = df_selected.drop(columns='max_z')
    
            # Sort the DataFrame by the 'z' column in ascending order
            df_selected_sorted = df_selected.sort_values(by='z', ascending=True)
            #if (df_selected.equals(df_selected_sorted)) == False:
            #    for x,y,i,j in zip(df_selected[var],df_selected_sorted[var],df_selected[var].index,df_selected_sorted[var].index):
            #        #print ('ind', i)
            #        if x != y:
            #            print (x,y,i,j)#,df_selected.iloc[i],df_selected_sorted.iloc[i])
    
            # Round the 'z' values and calculate the mean for each group
            df_selected_sorted['zr'] = df_selected_sorted['z'].round()
            #print (df_selected_sorted['zr'])
    
            #print (df_selected_sorted)
            df_selected_sorted = df_selected_sorted.groupby('zr').mean()
            #print (df_selected_sorted)
    
            # fill gaps in data
            df_selected_tmp = pd.DataFrame({'zr': np.arange(min(df_selected_sorted.reset_index()['zr']),max(df_selected_sorted.reset_index()['zr'])+1,1)})
            df_selected_sorted = pd.merge(df_selected_tmp, df_selected_sorted, how='outer', on='zr') #.fillna(0)
            df_selected_sorted = df_selected_sorted.interpolate(method='linear', axis=0)
    
            # Calculate the rolling mean of the sorted DataFrame
            if len(df_selected_sorted) != 1:
                df_selected_sorted_smoothed = df_selected_sorted.rolling(3).mean()
            else:
                df_selected_sorted_smoothed = df_selected_sorted
            #print (df_selected_sorted_smoothed)
            if df_tmp == '13-13-paraglider':
                print (df_selected)
            # Plotting temp vs z
            ax1.plot(df_selected[var], df_selected['z'], marker='o', ls='-', lw=1, ms=5, label=df_tmp)#imets[df_tmp]['lat'].iloc[0])
            ax2.plot(df_selected_sorted_smoothed[var], df_selected_sorted_smoothed['z'], marker='o', ls='-', lw=1, ms=5, label=df_tmp)#imets[df_tmp]['lat'].iloc[0])
            #plt.savefig(loc_fig)
    
            # Add the processed DataFrame to the list
            if len(df_selected_sorted) != 1:
                processed_dfs.append(df_selected_sorted_smoothed[2:].reset_index(drop=True))  #first two rows are nans due to smoothing 
            else:
                processed_dfs.append(df_selected_sorted_smoothed.reset_index(drop=True))
            
    if t[:5] == '12-11':
        ax1.plot(RS2.loc[:RS2_ind,var], RS2.loc[:RS2_ind,'z'], c='k', marker='o', ls='-', lw=1, ms=5, label='radiosonde')
        ax2.plot(RS2.loc[:RS2_ind,var], RS2.loc[:RS2_ind,'z'], c='k', marker='o', ls='-', lw=1, ms=5, label='radiosonde')
    if t[:5] == '13-5-':
        ax1.plot(RS4.loc[:RS4_ind,var], RS4.loc[:RS4_ind,'z'], c='k', marker='o', ls='-', lw=1, ms=5, label='radiosonde')
        ax2.plot(RS4.loc[:RS4_ind,var], RS4.loc[:RS4_ind,'z'], c='k', marker='o', ls='-', lw=1, ms=5, label='radiosonde')
    
    if var == 't':
        ax1.plot(ttNB.loc[ttNB['date_N5_']==date,['t_N5_']], glacier_aws[2]+2.79, c='y', marker='o', ls='-', lw=1, ms=2*5)
        ax1.plot(ttNB.loc[ttNB['date_N4_']==date,['t_N4_']], glacier_aws[2]+1.79, c='y', marker='o', ls='-', lw=1, ms=2*5)
        ax1.plot(ttNB.loc[ttNB['date_N3_']==date,['t_N3_']], glacier_aws[2], c='y', marker='o', ls='-', lw=1, ms=2*5)
        #ax1.plot(ttNB.loc[ttNB['date_N3_']==date,['t_N3_']], glacier_aws[2], c='y', marker='o', ls='-', lw=1, ms=2*5, label='tinytags')
        ax1.scatter(humilog_upper.loc[humilog_upper['date']>=date,'T'].iloc[0],upper_tongue[2], c='turquoise', s=200, marker='^', zorder=1000)
        ax1.scatter(humilog_lower.loc[humilog_upper['date']>=date,'T'].iloc[0],lower_tongue[2], c='orange', s=200, marker='^', zorder=1000)
        ax1.scatter(FF.loc[FF['date']>=date,'T'].iloc[0],inlet_aws[2], c='grey', marker='^', s=200, zorder=1000)
        ax1.scatter(NV.loc[NV['date']>=date,'t'].iloc[0],outlet_aws[2], marker='^', s=200, zorder=1000)
        ax2.plot(ttNB.loc[ttNB['date_N5_']==date,['t_N5_']], glacier_aws[2]+2.79, c='y', marker='o', ls='-', lw=1, ms=2*5)
        ax2.plot(ttNB.loc[ttNB['date_N4_']==date,['t_N4_']], glacier_aws[2]+1.79, c='y', marker='o', ls='-', lw=1, ms=2*5)
        ax2.plot(ttNB.loc[ttNB['date_N3_']==date,['t_N3_']], glacier_aws[2], c='y', marker='o', ls='-', lw=1, ms=2*5)
        #ax2.plot(ttNB.loc[ttNB['date_N3_']==date,['t_N3_']], glacier_aws[2], c='y', marker='o', ls='-', lw=1, ms=2*5, label='tinytags')
        ax2.scatter(humilog_upper.loc[humilog_upper['date']>=date,'T'].iloc[0],upper_tongue[2], c='turquoise', s=200, marker='^', zorder=1000)
        ax2.scatter(humilog_lower.loc[humilog_upper['date']>=date,'T'].iloc[0],lower_tongue[2], c='orange', s=200, marker='^', zorder=1000)
        ax2.scatter(FF.loc[FF['date']>=date,'T'].iloc[0],inlet_aws[2], c='grey', marker='^', s=200, zorder=1000)
        ax2.scatter(NV.loc[NV['date']>=date,'t'].iloc[0],outlet_aws[2], marker='^', s=200, zorder=1000)
    
    ax1.set_title(f'{date} not smoothed')
    ax1.legend(loc='best')
    ax2.set_title(f'{date} smoothed')
    ax2.legend(loc='best')
    
    #plt.savefig(f'plots/profiles_{var}_{t[:5]}.png')
    
    z_dist = []
    for i in range(len(processed_dfs)):
        if processed_dfs[i].empty:
            print(f'DataFrame {[i]} is empty!')
        else:
            z_dist.extend([np.min(processed_dfs[i]['zr'])])
    
    # Define a list of distances
    dist = intdist[0::10] #list(round(hor_up.loc[hor_up['ascents'] != 'nan','x']))### distance along centerline
    dist = [round(x) for x in dist]
    
    # Adding a 'dist' column to each DataFrame with corresponding values from the 'dist' list
    for i in range(len(processed_dfs)):
        processed_dfs[i]['dist'] = dist[i]
    
    #if var == 'pt':
        
    fig, ax2 = plt.subplots(nrows=1, figsize=(10, 6))
    plt.style.use('seaborn-v0_8-talk')
    
    # the 'z' needs to be shifted so that the lowest (first) value is the value of intalts for the corresponding location
    
    def shift_z(df):
        d = df['dist'].iloc[0]
        idx = np.abs(intdist - d).argmin()
        z_base = intalts[idx]
        return df['z'] - df['z'].min() + z_base
    
    cntr1 = plt.plot(intdist, intalts)
    for i in range(len(processed_dfs)):
        if shift_z(processed_dfs[i])[0] > processed_dfs[i]['z'][0]:
            print (f'shifting profile nr {i} up {shift_z(processed_dfs[i])[0] - processed_dfs[i]['z'][0]} meters')
            processed_dfs[i]['z_shifted'] = shift_z(processed_dfs[i])
        else:
            # not shifting down
            processed_dfs[i]['z_shifted'] = processed_dfs[i]['z']
        cntr1 = plt.plot(processed_dfs[i]['dist'], processed_dfs[i]['z_shifted'])
    
    ax2.set_xlabel('distance along centerline [m]')
    ax2.set_ylabel('elevation [m a.s.l.]')
    ax2.grid()
    
    #z_start = [np.min(processed_dfs[i]['z_shifted']) for i in range(len(processed_dfs))]
    z_end   = [np.max(processed_dfs[i]['z_shifted']) for i in range(len(processed_dfs))]
    start_level = 195
    end_level = np.max(z_end)
    x_plot = interpolate_array(intdist, len(intdist), 10)
    z_plot = np.arange(start_level, end_level+1, 1)
    xdim = len(x_plot)
    zdim = len(z_plot)
    xs, zs = np.meshgrid(x_plot, z_plot)
    
    # creating empty 2D array (xyspacenew) for interpolation purposes
    # first entry of each column is ground level (not a specific altitude level)
    xyspacenew = np.ones((xdim, zdim,))*np.nan
    
    # populating the xyspacenew array with variable values from each df
    for idx, df in enumerate(processed_dfs):
        row_index = idx*100
        start_index = int(np.nanmin(df['z_shifted'])-intalts[idx*10]) # set profile position relative to ground level
        if start_index < 0:
            print ('profile shifted up to ground level')
            start_index = 0 # ensure all profiles start at or above ground
        end_index = start_index+df['z_shifted'].count()
        print (start_index,end_index)
        xyspacenew[row_index, start_index:end_index] = df[var]  # assign t/pt/rh values to the array from bottom of profile and up
        
    xyspacenew = xyspacenew.T
    
    # add outlet/inlet near surface observation if missing
    ex = '-' # extra string needed for imet keys with 2 digit hour (i.e., 10, 13, 16)
    if t[4] == '-':
        ex = ''
        
    add_glacier_obs = True
    add_inlet_obs = True
    add_outlet_obs = True
    
    
    # locations along interpolation line
    closest_point_outlet = min(intpts, key=lambda x: pt_out.distance(Point(x)))
    xind_outlet = intdist[intpts.index(closest_point_outlet)]
    closest_point_inlet = min(intpts, key=lambda x: pt_inl.distance(Point(x)))
    xind_inlet = intdist[intpts.index(closest_point_inlet)]
    print (xind_inlet)
    closest_point_front = min(intpts, key=lambda x: pt_fro.distance(Point(x)))
    xind_front = intdist[intpts.index(closest_point_front)]
    closest_point_tonlow = min(intpts, key=lambda x: pt_tlo.distance(Point(x)))
    xind_tonlow = intdist[intpts.index(closest_point_tonlow)]
    closest_point_tonup = min(intpts, key=lambda x: pt_tup.distance(Point(x)))
    xind_tonup = intdist[intpts.index(closest_point_tonup)]
    closest_point = min(intpts, key=lambda x: pt_gla.distance(Point(x)))
    xind_glacier = intdist[intpts.index(closest_point)]
    
    if var == 't':
        try:
            imets[t[:5]+ex+'gla-lower']
        except:
            pass
        else:
            if np.isnan(xyspacenew[2,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100]):
                if add_glacier_obs == True:
                    print ('adding glacier near-surface observation and interpolating linearly up to start of vertical profile')
                    first_nonnan = np.where(np.isnan(xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100])==False)[0][0]
                    #print (xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100])
                    xyspacenew[1,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100] = ttNB.loc[ttNB['date_N3_']==date,'t_N3_']
                    #if np.isnan(xyspacenew[1,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100]):
                    xyspacenew[2,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100] = ttNB.loc[ttNB['date_N4_']==date,'t_N4_']
                    #    if np.isnan(xyspacenew[2,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100]):
                    xyspacenew[3,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100] = ttNB.loc[ttNB['date_N5_']==date,'t_N5_']
                            #print (xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100])
                    first_nan = np.where(np.isnan(xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100]))[0][0]
            
                    x = np.array([first_nan-1, first_nonnan])#, 1)
                    y = xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100]#[first_nan-1:first_nonnan+1]
                    y = np.array([y[first_nan-1],y[first_nonnan]])
                    f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
                    x_new = np.arange(first_nan, first_nonnan, 1)
                    xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100][first_nan:first_nonnan] = f(x_new)
            #        print (xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'gla-lower')*100][first_nan:first_nonnan])
                else:
                    print ('consider adding glacier observation because profile starts above surface')
        try:
            imets[t[:5]+ex+'out']
        except:
            pass
        else:
            if np.isnan(xyspacenew[0,list(imets.keys()).index(t[:5]+ex+'out')*100]): #imets[t[:5]+'out']['z'][0] > outlet_aws[2]+10:
                if add_outlet_obs == True:
                    print ('adding outlet near-surface observation and interpolating linearly up to start of vertical profile')
                    first_nonnan = np.where(np.isnan(xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'out')*100])==False)[0][0]
                    #print (xyspacenew[:40,list(imets.keys()).index(t[:5]+ex+'out')*100])
                    #xyspacenew[0,list(imets.keys()).index(t[:5]+ex+'inl')*100] = 
                    xyspacenew[0,list(imets.keys()).index(t[:5]+ex+'out')*100] = NV.loc[NV['date']>=date,'t'].iloc[0]
                    #print (xyspacenew[:40,list(imets.keys()).index(t[:5]+ex+'out')*100])
                    first_nan = np.where(np.isnan(xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'out')*100]))[0][0]
            
                    x = np.array([first_nan-1, first_nonnan])#, 1)
                    y = xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'out')*100]#[first_nan-1:first_nonnan+1]
                    y = np.array([y[first_nan-1],y[first_nonnan]])
                    f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
                    x_new = np.arange(first_nan, first_nonnan, 1)
                    xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'out')*100][first_nan:first_nonnan] = f(x_new)
                    #print (x, y, x_new, f(x_new))
                    #print (xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'out')*100][first_nan:first_nonnan])
                    #print (xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'inl')*100])
                else:
                    print ('adding outlet observation without interpolating up')
                    xyspacenew = add_obs(xyspacenew, xs, zs, 'outlet', xind_outlet, var=var)
    if var == 't' or var == 'rh':
        try:
            imets[t[:5]+ex+'inl']
        except:
            print ('adding inlet observation without interpolating up')
            xyspacenew = add_obs(xyspacenew, xs, zs, 'inlet', xind_inlet, var=var)
        else:
            if np.isnan(xyspacenew[0,list(imets.keys()).index(t[:5]+ex+'inl')*100]):
            #if imets[t[:5]+'inl']['z'][0] > inlet_aws[2]+10:
                if add_inlet_obs == True:
                    try:
                        first_nonnan = np.where(np.isnan(xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'inl')*100])==False)[0][0]
                    except:
                        print ('adding inlet observation without interpolating up')
                        xyspacenew = add_obs(xyspacenew, xs, zs, 'inlet', xind_inlet, var=var)
                    else:
                        print ('adding inlet near-surface observation and interpolating linearly up to start of vertical profile')
                        #print (xyspacenew[:20,list(imets.keys()).index(t[:5]+ex+'inl')*100])
                        #xyspacenew[0,list(imets.keys()).index(t[:5]+ex+'inl')*100] = 
                        if var == 't':
                            xyspacenew[0,list(imets.keys()).index(t[:5]+ex+'inl')*100] = FF.loc[FF['date']>=date,'T'].iloc[0]
                        elif var == 'rh':
                            xyspacenew[0,list(imets.keys()).index(t[:5]+ex+'inl')*100] = FF.loc[FF['date']>=date,'RH'].iloc[0]
                        #xyspacenew = add_obs(xyspacenew, xs, zs, 'inlet')
                        #print (xyspacenew[:20,list(imets.keys()).index(t[:5]+ex+'inl')*100])
                        first_nan = np.where(np.isnan(xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'inl')*100]))[0][0]
                
                        x = np.array([first_nan-1, first_nonnan])#, 1)
                        y = xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'inl')*100]#[first_nan-1:first_nonnan+1]
                        y = np.array([y[first_nan-1],y[first_nonnan]])
                        f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
                        x_new = np.arange(first_nan, first_nonnan, 1)
                        xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'inl')*100][first_nan:first_nonnan] = f(x_new)
                        #print (xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'inl')*100][first_nan:first_nonnan])
                        #print (xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'inl')*100])
                else:
                    print ('adding inlet observation without interpolating up')
                    xyspacenew = add_obs(xyspacenew, xs, zs, 'inlet', xind_inlet, var=var)
            
    #        print ('adding outlet observation because profile starts more than 10 m above surface')
    #        print (xyspacenew[:,list(imets.keys()).index(t[:5]+ex+'out')*100])
    
    # adding humilog observations
    #print (xyspacenew[0:5,intpts.index(closest_point_tonlow)*10])
    if var == 't':
        xyspacenew[1,intpts.index(closest_point_tonlow)*10] = humilog_lower.loc[humilog_lower['date']>=date,'T'].iloc[0]
        xyspacenew[1,intpts.index(closest_point_tonup)*10] = humilog_upper.loc[humilog_upper['date']>=date,'T'].iloc[0]
    elif var == 'rh':
        xyspacenew[1,intpts.index(closest_point_tonlow)*10] = humilog_lower.loc[humilog_lower['date']>=date,'RH'].iloc[0]
        xyspacenew[1,intpts.index(closest_point_tonup)*10] = humilog_upper.loc[humilog_upper['date']>=date,'RH'].iloc[0]
    #print (xyspacenew[0:5,intpts.index(closest_point_tonlow)*10])
    
    
    # setting the surface temperature equal to 0 deg over the melting glacier
    xyspacenew[0,np.where(x_plot == xind_front)[0][0]+1:] = 0
    # setting the surface temperature equal to 3 deg over the proglacial lake (average lake temp during IOP)
    xyspacenew[0,1:np.where(x_plot == xind_inlet)[0][0]] = 4
    
    
    
    # Interpolate NaN values in the xyspacenew array
    for row in xyspacenew:
        mask = np.isnan(row)   # identify nans
        x = np.where(~mask)[0] # first value on row without nan
        y = row[~mask]         # values on row without nan
        #print (row, y, x)
        if len(x) > 1:  # Make sure there are at least two points to interpolate
            f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            x_new = np.where(mask)[0] # Only interpolate points that are NaN
            x_new_valid = x_new[(x_new >= min(x)) & (x_new <= max(x))] # only interpolate points within the range of x
            row[mask] = np.nan  # Set all masked values to NaN for now
            row[x_new_valid] = f(x_new_valid)  # Only interpolate valid points
        else:
            #If there are fewer than two points, leave the NaN values as they are
            continue
    
    # shift columns up to altitude level
    for idx, col in enumerate(xyspacenew.T):
        z_shift = int(intalts_extended[idx])-start_level
        col[z_shift:] = col[:-z_shift]
        col[:z_shift] = np.nan
    
    
    # prepare plot
    
    cmap = plt.cm.coolwarm#RdBu_r#viridis#
    if var == 't':
        vmin, vmax = 278,284
        cblab = 'air temperature (\u00b0C)'#K)'#
    elif var == 'rh':
        vmin, vmax = 30,90
        cblab = 'relative humidity (%)'
        cmap = plt.cm.RdBu
    elif var == 'pt':
        vmin, vmax = 278, 288 #8,14
        cblab = 'potential temperature (K)'#\u00b0C)'
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 16)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    
    add = 0
    if var == 't':
        add = 273.15
        
    def add_observations(ax):
        for df, alt in zip(imets, intalts_extended[::100]):
            df = imets[df]
            if np.min(df['z']) < alt: # shift profiles up if they start too low
                df['z'] += alt-np.min(df['z'])
                print (np.min(df['z']), alt, intalts_extended[-1]-np.min(df['z']))
            ax.scatter(df['xind'][::3],df['z'][::3],c=df[var][::3]+add, ec='k', marker='o', s=10, cmap=cmap, norm=norm, zorder=900)
            scatter = ax.scatter(df['xind'][::3],df['z'][::3],c=df[var][::3]+add, ec='none', marker='o', s=8, cmap=cmap, norm=norm, zorder=1000)
            #dfxind,dfz = np.meshgrid(df['xind'],df['z'])
            #plt.pcolormesh(dfxind,dfz,df[var])
        
    #    if t[:5] == '12-11':       
    #        ax.scatter(np.ones(len(RS2.loc[:RS2_ind]))*xind_front, RS2.loc[:RS2_ind,'z'], c=RS2.loc[:RS2_ind,var]+add, ec='k', marker='s', s=12, cmap=cmap, norm=norm, zorder=900)
    #        scatter = ax.scatter(np.ones(len(RS2.loc[:RS2_ind]))*xind_front, RS2.loc[:RS2_ind,'z'], c=RS2.loc[:RS2_ind,var]+add, ec='none', marker='s', s=10, cmap=cmap, norm=norm, zorder=1000)
    #    if t[:5] == '13-5-':       
    #        ax.scatter(np.ones(len(RS4.loc[:RS4_ind]))*xind_front, RS4.loc[:RS4_ind,'z'], c=RS4.loc[:RS4_ind,var]+add, ec='k', marker='s', s=12, cmap=cmap, norm=norm, zorder=900)
    #        scatter = ax.scatter(np.ones(len(RS4.loc[:RS4_ind]))*xind_front, RS4.loc[:RS4_ind,'z'], c=RS4.loc[:RS4_ind,var]+add, ec='none', marker='s', s=10, cmap=cmap, norm=norm, zorder=1000)
            
        if var == 't':
            sc1 = ax.scatter(xind_glacier,glacier_aws[2]+2.79,c=ttNB.loc[ttNB['date_N5_']==date,'t_N5_']+add, ec='k', marker='o', s=100, cmap=cmap, norm=norm, zorder=1000)
            sc2 = ax.scatter(xind_glacier,glacier_aws[2]+1.79,c=ttNB.loc[ttNB['date_N3_']==date,'t_N3_']+add, ec='k', marker=MarkerStyle('o', fillstyle='bottom'), s=100, cmap=cmap, norm=norm, zorder=1100)
            #plt.scatter(xind_glacier,glacier_aws[2],c=ttNB.loc[ttNB['date_N3_']==date,'t_N3_'], ec='k', marker='o', s=10, cmap=cmap, norm=norm, zorder=1000)
            #plt.scatter(xind_glacier,glacier_aws[2],c=np.mean([NB.loc[NB['date']<date, 't_u'].iloc[-1],NB.loc[NB['date']>date, 't_u'].iloc[0]]), ec='k', marker='^', s=200, cmap=cmap, norm=norm, zorder=1000)
            if t[:2] == '13':
                sc3 = ax.scatter(x_plot[0],outlet_aws[2]+5,c=NV.loc[NV['date']>=date,'t'].iloc[0]+add, ec='k', marker='o', s=100, cmap=cmap, norm=norm, zorder=1000)
            else:
                sc3 = sc2
        if var == 't':
            var2 = 'T'
        elif var == 'rh':
            var2 = 'RH'
        ax.scatter(xind_tonup,intalts[intpts.index(closest_point_tonup)],c=humilog_upper.loc[humilog_upper['date']>=date,var2].iloc[0]+add, ec='k', s=50, marker='o', cmap=cmap, norm=norm, zorder=1000)
        ax.scatter(xind_tonlow,intalts[intpts.index(closest_point_tonlow)],c=humilog_lower.loc[humilog_upper['date']>=date,var2].iloc[0]+add, ec='k', s=50, marker='o', cmap=cmap, norm=norm, zorder=1000)
        sc4 = ax.scatter(xind_inlet,intalts[intpts.index(closest_point_inlet)],c=FF.loc[FF['date']>=date,var2].iloc[0]+add, ec='k', marker='o', s=100, cmap=cmap, norm=norm, zorder=1000)
        return ax, sc1, sc2, sc3, sc4
    
    
    # add ice thickness
    df = pd.read_excel('data/nigard_ice.xlsx')
    df.columns = ['lon', 'lat', 'elevation', 'thickness']
    
    intpts_extended = interpolate_array(intpts, len(intpts), 10, 'coord')
    
    from scipy.spatial import KDTree
    coordinates = df[['lon', 'lat']].values
    tree = KDTree(coordinates)
    
    thickness_array = []
    for coord in intpts_extended:
        dis, idx = tree.query(coord)
        thickness = df['thickness'].iloc[idx]
        thickness_array.append(thickness)
    thickness_array = np.array(thickness_array)
    thickness_array[thickness_array < 5] = 0
    
    
    # estimate point on interpolation line for all UAV measurements in imets:
    for df in imets:
        df = imets[df]
        df['xind'] = df['z']
        for i in range(len(df)):
            closest_point = min(intpts, key=lambda x: (Point(df.loc[i]['lat'],df.loc[i]['lon'])).distance(Point(x)))
            df.loc[i,'xind'] = intdist[intpts.index(closest_point)]
    
    if var == 'pt':
        pt_contours = xyspacenew
    
    
    # plot
    
    ff=.8
    
    if t[:2] == '13':
        fig,ax1 = plt.subplots(1, figsize=(8*3/4*ff*.84, 6*ff), dpi=200)
    elif t[:2] == '14':
        fig,ax1 = plt.subplots(1, figsize=(5*32/54*ff*1.08, 6.3*ff), dpi=200)
    
    plt.style.use('seaborn-v0_8-talk')
    #plt.plot(x_plot, intalts_extended, c='k', zorder=100)
    ax1.fill_between(x_plot, intalts_extended, start_level, linestyle='-', color='k', zorder=10)
    ax1.fill_between(x_plot, intalts_extended-thickness_array, intalts_extended, linestyle='-', color='paleturquoise', zorder=10)
    ax1.fill_between([-30,0], [intalts_extended[0], intalts_extended[0]], start_level, linestyle='-', color='k', zorder=10)
    ax1.fill_between([-30,0], [intalts_extended[0]-thickness_array[0], intalts_extended[0]-thickness_array[0]], [intalts_extended[0], intalts_extended[0]], linestyle='-', color='paleturquoise', zorder=10)
    ax1.fill_between([5526, 5558], 0, intalts_extended[-1], linestyle='-', color='k', zorder=10)
    ax1.fill_between([5526, 5558], intalts_extended[-1]-thickness_array[-1], intalts_extended[-1], linestyle='-', color='paleturquoise', zorder=10)

    print (var)
    if var == 't' or var == 'rh':
        cntr1 = ax1.contourf(xs,zs,xyspacenew+add, cmap=cmap, norm=norm, levels=levels, extend='both', zorder=800)
        ax1, sc1, sc2, sc3, sc4 = add_observations(ax1)
        sc1.set_clip_on(False)
        sc2.set_clip_on(False)
        sc3.set_clip_on(False)
        sc4.set_clip_on(False)
        c2 = ax1.contour(xs, zs, gaussian_filter(pt_contours, sigma=3), colors='k', linewidths=1, levels=np.arange(270,300,1), zorder=900)#270,300,1), zorder=900)
        #c2 = ax1.contour(xs, zs, gaussian_filter(pt_contours, sigma=3)-273.15, colors='k', linewidths=1, levels=np.arange(0,12,1), zorder=900)#270,300,1), zorder=900)
        manual_loc = [(1,500),(200,200),(800,800),(1200,700)]
        #ax1.clabel(c2, inline=True, fontsize=12)
    if var == 'pt':
        cntr1 = ax1.contourf(xs, zs, gaussian_filter(pt_contours, sigma=3), cmap=cmap, norm=norm, levels=15, zorder=800)
    #cntr1 = plt.pcolormesh(xs,zs,xyspacenew, cmap=cmap, norm=norm, zorder=1000) #"RdBu_r"
    #cntr1 = plt.pcolormesh(xs,zs,gaussian_filter(xyspacenew, sigma=1), cmap=cmap, norm=norm, zorder=1000) #"RdBu_r"
    
    # add lines for lake (and glacier?)
    try:
        ind = list(imets.keys()).index(t[:5]+ex+'inl')*100
    except:
        try:
            ind = list(imets.keys()).index(t[:5]+ex+'inlet')*100
        except:
            print ('cannot find lake inlet')
            ind = np.where(x_plot == xind_inlet)[0][0]
            ax1.plot(x_plot[:ind-5], [i-5 for i in intalts_extended[:ind-5]], c='turquoise', lw=5, alpha=.6, zorder=500)
        else:
            ax1.plot(x_plot[:ind-5], [i-5 for i in intalts_extended[:ind-5]], c='turquoise', lw=5, alpha=.6, zorder=500)
            print ('plotting lake line')
    else:
        ax1.plot(x_plot[:ind-5], [i-5 for i in intalts_extended[:ind-5]], c='turquoise', lw=5, alpha=.6, zorder=500)
        print ('plotting lake line')
    try:
        ind = list(imets.keys()).index(t[:5]+ex+'fro')*100
    except:
        print ('cannot find front')
        ind = xind_front
        ax1.plot(x_plot[ind+shift:], [i-5 for i in intalts_extended[ind+shift:]], c='turquoise', lw=5)
    else:
        shift = 10
        #ax1.plot(x_plot[ind+shift:], [i-8 for i in intalts_extended[ind+shift:]], c='lightgrey', lw=5, zorder=500)
        print ('plotting glacier line')
    
    #cb = plt.colorbar(cntr1, ax=ax1, extend='both', orientation='horizontal')#'vertical')#
    #cb.set_label(cblab)#, rotation=270, labelpad=15)
    #cb.set_ticks(levels[::5])
    
    # add labels for locations
    fs = 14
    ax1.text(xind_glacier-130, 203+35, 'G', color=c[9], ha='center', va='bottom', rotation=0, fontsize=fs, zorder=1000, 
            bbox=dict(facecolor='none', edgecolor='none', lw=1.5))
    if t[:2] == '13':
        ax1.text(dist[-2], 203+35, 'F', color=c[7], ha='center', va='bottom', rotation=0, fontsize=fs, zorder=1000, 
                bbox=dict(facecolor='none', edgecolor='none', lw=1.5))
    elif t[:2] == '14':
        ax1.text(xind_front, 203+35, 'F', color=c[7], ha='center', va='bottom', rotation=0, fontsize=fs, zorder=1000, 
                bbox=dict(facecolor='none', edgecolor='none', lw=1.5))
    if t[:5] == '13-5-':
        ax1.text(dist[1]*0.7, 203+35, 'I', color=c[1], ha='center', va='bottom', rotation=0, fontsize=fs, zorder=1000, 
                bbox=dict(facecolor='none', edgecolor='none', lw=1.5))
    elif t[:2] == '13':
        ax1.text(dist[-3], 203+35, 'I', color=c[1], ha='center', va='bottom', rotation=0, fontsize=fs, zorder=1000, 
                bbox=dict(facecolor='none', edgecolor='none', lw=1.5))
    elif t[:2] == '14':
        ax1.text(xind_inlet+70, 203+35, 'I', color=c[1], ha='center', va='bottom', rotation=0, fontsize=fs, zorder=1000, 
                bbox=dict(facecolor='none', edgecolor='none', lw=1.5))
    if t[:5] == '13-13':
        ax1.text(dist[1], 203+35, 'L', color='y', ha='center', va='bottom', rotation=0, fontsize=fs, zorder=1000, 
                bbox=dict(facecolor='none', edgecolor='none', lw=1.5))
    if t[:2] == '13':
        ax1.text(dist[0]+140, 203+35, 'O', color=c[6], ha='center', va='bottom', rotation=0, fontsize=fs, zorder=100, #dist[0]+70
                bbox=dict(facecolor='none', edgecolor='none', lw=1.5))
    
    ax1.set_xlabel('distance along centerline (km)')
    if t[:2] == '14':
        ax1.set_xlabel('dist. along centerline (km)')
    ax1.set_ylabel('elevation (m a.s.l.)')
    ax1.set_xlim(x_plot[0]-30,x_plot[-1]+30)
    if t[:5] == '14-5-' or t[:5] == '14-13':
        ax1.set_xlim(xind_inlet-30,x_plot[-1]+30)
    elif t[:5] == '14-7-' or t[:5] == '14-10':
        ax1.set_xlim(xind_inlet-30,xind_glacier+30)
    ax1.invert_xaxis()
    if t[:2] == '14':
        ax1.set_xticks(np.arange(ax1.get_xlim()[0]-30-2000,ax1.get_xlim()[0]+1,1000)) #ax1.get_xlim()[0]-80-5000,
        ax1.set_xticklabels(np.flip(np.arange(0,3,1)))
    else:
        ax1.set_xticks(np.arange(ax1.get_xlim()[0]-30-5000,ax1.get_xlim()[0]+1,1000)) #ax1.get_xlim()[0]-80-5000,
        ax1.set_xticklabels(np.flip(np.arange(0,6,1)))
    ax1.fill_between([-30,0], [intalts_extended[0],intalts_extended[0]], start_level, linestyle='-', color='k', zorder=100) # repair edges
    if t[:2] == '13' or t[:5] == '14-5-':
        ax1.fill_between([xind_glacier-30,ax1.get_xlim()[0]], [intalts_extended[-1],intalts_extended[-1]], start_level, linestyle='-', color='k', zorder=100) # repair edges
    #    ax1.fill_between([xind_glacier-33,xind_glacier-5], [intalts_extended[-1]-thickness_array[-1]], [intalts_extended[-1]], linestyle='-', color='paleturquoise', zorder=110)
        ax1.fill_between([xind_glacier-39,xind_glacier+25], [intalts_extended[-1]-thickness_array[-1]], [intalts_extended[-1]], linestyle='-', color='paleturquoise', zorder=110)
    #elif t[:2] == '14':
    #    ax1.fill_between([xind_glacier-53,xind_glacier+25], [intalts_extended[-1]-thickness_array[-1]], [intalts_extended[-1]], linestyle='-', color='paleturquoise', zorder=110)
    ax1.set_ylim(start_level+40,730)#940#np.max(zs))
    
    if t[3:4] == '5':
        lab = 'a'
    elif t[3:5] == '10':
        lab = 'b'    
    elif t[3:5] == '13':
        lab = 'c'
    
    if t[:2] == '13':
        ax1.set_title(f'{lab})     {str(date)[8:10]} Sept. {str(date)[-8:-3]}       ')
    else:
        ax1.set_title(f'{lab}) {str(date)[8:10]} Sept.\n{str(date)[-8:-3]}')
    
    
    plt.tight_layout()
    #plt.savefig(f'plots/colorbar.png')
    if var == 't':
        plt.savefig(f'plots/{var}_{t[:5]}_linear.pdf')
        plt.show()

    elif var=='pt':
        return pt_contours



# --------------------------------------------------------------------
# plot surface data from wrf and observations
# --------------------------------------------------------------------

def findClosestGridPoints(data, lat=61.659, lon=7.276):

    sn_red = 45
    we_red = 50

    #la,lo = wrf.ll_to_xy(data,lat,lon)[::-1].values
    if lat==61.659 and lon==7.276:
        la,lo = 48,78 #57,85
    elif lat==61.687 and lon==7.197:
        la,lo = 57, 65
    elif lat==61.67595235467888 and lon==7.211611010507808:
        la,lo = 53, 68 #54,67
    elif lat==61.6672661 and lon==7.2415509:
        la,lo = 50,72
    la -= sn_red
    lo -= we_red
        
    
    if   ((data['XLAT'][0][la,lo] < lat) & (data['XLONG'][0][la,lo] < lon)):
        #print ('<<', la, lo, la+1, lo+1)
        las = [la,la,la+1,la+1]; los = [lo,lo+1,lo+1,lo]
    elif ((data['XLAT'][0][la,lo] > lat) & (data['XLONG'][0][la,lo] < lon)):
        #print ('><', la, lo, la+1, lo-1)
        las = [la,la,la-1,la-1]; los = [lo,lo+1,lo+1,lo]
    elif ((data['XLAT'][0][la,lo] > lat) & (data['XLONG'][0][la,lo] > lon)):
        #print ('>>', la, lo, la-1, lo-1)
        las = [la,la,la-1,la-1]; los = [lo,lo-1,lo-1,lo]
    elif ((data['XLAT'][0][la,lo] < lat) & (data['XLONG'][0][la,lo] > lon)):
        #print ('<>', la, lo, la-1, lo+1)
        las = [la,la,la+1,la+1]; los = [lo,lo-1,lo-1,lo]
        
    #print ('coordinates of station', (lat,lon))
    distances = []
    for la,lo in zip(las,los):
        distances.append(geopy.distance.geodesic((lat,lon),(data['XLAT'][0][la,lo],data['XLONG'][0][la,lo])).m)
        
    return (las,los,distances)

def InverseDistanceWeighted(lat,lon,data,var='T2'):
    #if len(data) == len(WRF400):
    #    #print ('using WRF400 data')
    #    #WRF_3D = Dataset('WRF_data/2021_july_400m/wrfuserout_d03_2021-07-01_00:00:00')
    #    WRF_3D = Dataset('WRF_data/2021_JJA/ssthigh/wrfuserout_d03_2021-07-01_00:00:00')
    #else:
    #    #print ('using WRF1000 data')
    #    WRF_3D = Dataset('WRF_data/2021_JJA/sstlow/wrfout_d03_2021-07-01_00:00:09')
    #    WRF_3D = Dataset('WRF_data/2021_JJA/sstlow/wrfout_d03_2021-06-01_00:00:00')
    las,los,dis = (findClosestGridPoints(data=WRF_3D, lat=lat,lon=lon))
    if var == 'UVMET':
        weighted_u = []
        weighted_v = []
        #print (len(data['bottom_top']))
        for k in range(len(data['bottom_top'])):
            var4pts_u = []
            var4pts_v = []
            for i,j in zip(los,las):
                #print (i,j)
                #moddata_u = 
                #moddata_v = 
                #data.loc[(data['bottom_top']==k)&(data['south_north']==j)&(data['west_east']==i),var].reset_index(drop=True)
                #print (np.shape(moddata))
                var4pts_u.append(UVMET[0,:,k,j,i])#moddata_u)
                var4pts_v.append(UVMET[1,:,k,j,i])#moddata_v)
            #print (var4pts_u)
            weighted_u.append( (var4pts_u[0]/dis[0]+var4pts_u[1]/dis[1]+var4pts_u[2]/dis[2]+var4pts_u[3]/dis[3])/(1/dis[0]+1/dis[1]+1/dis[2]+1/dis[3]))
            weighted_v.append( (var4pts_v[0]/dis[0]+var4pts_v[1]/dis[1]+var4pts_v[2]/dis[2]+var4pts_v[3]/dis[3])/(1/dis[0]+1/dis[1]+1/dis[2]+1/dis[3]))
            #print (np.shape(weighted_u))
        weighted = [weighted_u,weighted_v]
    elif var == 'T':
        weighted = []
        #print (len(data['bottom_top']))
        for k in range(len(data['bottom_top'])):
            var4pts = []
            for i,j in zip(los,las):
                moddata = data.loc[(data['bottom_top']==k)&(data['south_north']==j)&(data['west_east']==i),var].reset_index(drop=True)
                var4pts.append(moddata)
            #print (var4pts_u)
            weighted.append( (var4pts[0]/dis[0]+var4pts[1]/dis[1]+var4pts[2]/dis[2]+var4pts[3]/dis[3])/(1/dis[0]+1/dis[1]+1/dis[2]+1/dis[3]))
            print ('shape weighted: ', np.shape(weighted))
               
    else:
        var4pts = []
        for i,j in zip(los,las):
            #print (i,j)
            moddata = data.loc[(data['south_north']==j)&(data['west_east']==i),var].reset_index(drop=True)
            #print (np.shape(moddata))
            var4pts.append(moddata)
        weighted = (var4pts[0]/dis[0]+var4pts[1]/dis[1]+var4pts[2]/dis[2]+var4pts[3]/dis[3])/(1/dis[0]+1/dis[1]+1/dis[2]+1/dis[3])
    return weighted

    
def plotTemperatureEvolution():

#    WRF1000_ts = pd.read_csv('data/WRF/wrf_files/WRF1000_ts.csv')
#    WRF400_ts = pd.read_csv('data/WRF/wrf_files/WRF400_ts.csv')
#
#    WRF = xr.open_mfdataset('data/WRF/wrf_files/wrfout_d03_surface.nc')
#
#    WRF['Times'] = (WRF['Times'].astype(np.unicode_))
#    WRF = (WRF.to_dataframe()).reset_index()
#    WRF['Times'] = pd.to_datetime(WRF['Times'], format='%Y-%m-%d_%H:%M:%S')
#    WRF = WRF.rename(columns={"Times": "date"})
#    
#    WRF['station_id'] = ''
#    for s in range(len(WRF1000_ts['station_id'].unique())):
#        (WRF.loc[(WRF['south_north']==np.array(WRF1000_ts.loc[~WRF1000_ts.grid_j.eq(WRF1000_ts.grid_j.shift()),'grid_j'])[s]) & \
#         (WRF['west_east']==np.array(WRF1000_ts.loc[~WRF1000_ts.grid_i.eq(WRF1000_ts.grid_i.shift()),'grid_i'])[s]),'station_id']) = WRF1000_ts['station_id'].unique()[s]
#    
#    # express temperature in celcius for comparison with other datasets
#    WRF['T2'] -= 273.15
#    
#    # calculate wind speed and direction
#    WRF['WS'] = np.sqrt(WRF['U10']**2+WRF['V10']**2)
#    WRF['WD'] = (np.arctan2(WRF['V10']/WRF['WS'],WRF['U10']/WRF['WS'])*180/np.pi)
#    WRF['WD'] += 180
#    WRF['WD'] = 90 - WRF['WD']
#    WRF['WD'] = (WRF['WD']%360)
#
#    WRF400 = xr.open_mfdataset('data/WRF/wrf_files/wrfout_d04_surface_coldlake.nc')
#    
#    WRF400['Times'] = (WRF400['Times'].astype(np.unicode_))
#    WRF400 = (WRF400.to_dataframe()).reset_index()
#    WRF400['Times'] = pd.to_datetime(WRF400['Times'], format='%Y-%m-%d_%H:%M:%S')
#    WRF400 = WRF400.rename(columns={"Times": "date"})
#    
#    WRF400['station_id'] = ''
#    for s in range(len(WRF400_ts['station_id'].unique())):
#        (WRF400.loc[(WRF400['south_north']==np.array(WRF400_ts.loc[~WRF400_ts.grid_j.eq(WRF400_ts.grid_j.shift()),'grid_j'])[s])&(WRF400['west_east']==np.array(WRF400_ts.loc[~WRF400_ts.grid_i.eq(WRF400_ts.grid_i.shift()),'grid_i'])[s]), 'station_id']) = WRF400_ts['station_id'].unique()[s]
#    
#    # express temperature in celcius for comparison with other datasets
#    WRF400['T2'] -= 273.15
#    
#    # calculate wind speed and direction
#    WRF400['WS'] = np.sqrt(WRF400['U10']**2+WRF400['V10']**2)
#    WRF400['WD'] = (np.arctan2(WRF400['V10']/WRF400['WS'],WRF400['U10']/WRF400['WS'])*180/np.pi)
#    WRF400['WD'] += 180
#    WRF400['WD'] = 90 - WRF400['WD']
#    WRF400['WD'] = (WRF400['WD']%360)
    
#    WRF_3D_d04 = Dataset('data/WRF/wrf_files/wrfout_d04_3D_warm-lake.nc')
#    WRF_3D = WRF_3D_d04
    
    # figure for paper
    
    lw  = .3
    s   = 50
    alp = .3
    lw2 = 2.2
    c = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    
    locnames = ['valley', 'glacier', 'inlet', 'valley2', 'outlet']
    
    WRF1000_label = '1000 m'; WRF400_label = '333 m'
    
    start = datetime(2023,9,13)#00,1,1)#2024,1,1)#12)
    end   = datetime(2023,9,15)#00,9,30)#2024,1,31)#15)
    
    plt.rcParams.update({'font.size': 19})#22})
    f,ax = plt.subplots(figsize=(15,8.5))
    
    # observations
    #ax = plotTimeSeries(data=MET_hourly,station_id=MET_hourly['station_id'].unique()[:1],variable='t') #c[2]
    ax.set_ylabel('temperature (\u00b0C)')
    ax.scatter(NB['date'],NB['t_u'],s=s,c=c[9],ec='k',lw=lw,label='glacier',zorder=100)
    ax.scatter(FF['date'],FF['T'],s=s,c=c[1],ec='k',lw=lw,label='inlet',zorder=100)
    ax.scatter(NV['date'],NV['t'],s=s,c=c[6],ec='k',lw=lw,label='outlet',zorder=100)
    ax.scatter(MG['date'],MG['T'],s=s,c=c[2],ec='k',lw=lw,label='valley1',zorder=100)
    #ax.scatter(BH['date'],BH['t'],s=s,c=c[5],ec='k',lw=lw,label='valley2')
    #ax.plot(era5['time'],era5['t2m'][:,era5_1000_j[0],era5_1000_i[0]],lw=4,alpha=.2,c='k',label=f'ERA5',zorder=-999)# {loc}')
    
    ax.scatter((), (), marker='o', c='grey', lw=1.5, label='AWS')
    #line, = 
    ax.plot((), (), marker='o', c='lightgrey', markerfacecolor='w', lw=1.5, label='1000 m (1 pt)')
    ax.plot((), (), marker='o', c='grey', markerfacecolor='w', lw=1.5, label='333 m (1 pt)')
    ax.plot((), (), marker='s', c='grey', markerfacecolor='w', lw=1.5, label='333 m (4 pts)')
    
    handles1, labels1 = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles1[:4], labels1[:4], ncol=1, loc=2)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles1[4:8], labels1[4:8], ncol=1, loc=4)
    #fig.add_artist(legend2)
    
    ax.axvspan(xmin=datetime(2023,9,13), xmax=datetime(2023,9,14), ymin=0, ymax=1, facecolor='grey', alpha=0.1)
    
    # simulations
    stations = ['MG','NB','FF','NV']
    for st,loc,ln in zip([0,1,12,13],stations[:2]+[stations[-2]]+[stations[-1]],locnames[:3]+[locnames[-1]]):#stations):
        if st == 0: # valley
            ci = 2
        if st == 1: # glacier
            ci = 9
        elif st == 11: # valley2
            ci = 5
        if st == 12: # inlet
            ci = 1
        elif st == 13: # outlet
            ci = 6
#        lat = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_lat'].iloc[0]
#        lon = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_lon'].iloc[0]
        #if st == 12:
        #    print ('shifting lat, lon of inlet away from water point')
        #    print (lat, lon)
        #    lon -= .003; lat += .002
#        print (loc, ln)
#        if ln == 'valley' or ln == 'glacier':
#            weighted = InverseDistanceWeighted(lat=lat,lon=lon,data=WRF,var='T2')
#        else:
#            weighted = np.load(f'data/WRF/wrf_profiles-and-timeseries/surface_temp/T2_coldlake_{ln}.npy')
#        i,j = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_i'].unique()[0], WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_j'].unique()[0]
        #if st == 12:
        #    print ('shifting i,j of inlet away from water point')
        #    i -= 1; j += 1
#        modeldata = WRF.loc[(WRF['south_north']==j)&(WRF['west_east']==i),['date','T2']].reset_index(drop=True)
        modeldata = pd.DataFrame()
        modeldata['T2'] = np.load(f'data/WRF/wrf_temp_evolution/T2_coldlake_{loc}_1000m_1pt.npy')
        modeldata['date'] = np.load(f'data/WRF/wrf_temp_evolution/times_1000m.npy')
        ax.plot   (modeldata['date']+pd.Timedelta(hours=2),modeldata['T2'],c=c[ci], alpha=alp, zorder=10)
        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),modeldata['T2'],c='w',ec='none',zorder=10)
        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),modeldata['T2'],c='w',ec=c[ci], alpha=alp,label=f'{WRF1000_label} {ln} (1 pt)',zorder=10)
    
#        WRF_3D = WRF_3D_d04
        #if ln == 'valley' or ln == 'glacier':
        #    weighted = InverseDistanceWeighted(lat=lat,lon=lon,data=WRF400,var='T2')
        #else:
#        weighted = np.load(f'data/WRF/wrf_profiles-and-timeseries/inverse-distance-weighted/T2_coldlake_{loc}.npy')
#        i,j = WRF400_ts.loc[WRF400_ts['station_id']==loc,'grid_i'].unique()[0], WRF400_ts.loc[WRF400_ts['station_id']==loc,'grid_j'].unique()[0]
#        #if st == 12:
#        #    print ('shifting i,j of inlet away from water point')
#        #    i -= 1; j += 1
#        modeldata = WRF400.loc[(WRF400['south_north']==j)&(WRF400['west_east']==i),['date','T2']].reset_index(drop=True)
#        if st == 0:
#            #print ('valley: ', modeldata['date'].iloc[41:], modeldata['T2'].iloc[41:])
#            valley_temp = modeldata['T2'].iloc[41:]
#        elif st == 12:
#            #print ('inlet: ', modeldata['date'].iloc[41:], modeldata['T2'].iloc[41:])
#            inlet_temp = modeldata['T2'].iloc[41:]
        modeldata = pd.DataFrame()
        modeldata['T2'] = np.load(f'data/WRF/wrf_temp_evolution/T2_coldlake_{loc}_333m_1pt.npy')
        modeldata['date'] = np.load(f'data/WRF/wrf_temp_evolution/times_333m.npy')
        ax.plot   (modeldata['date']+pd.Timedelta(hours=2),modeldata['T2'],c=c[ci], alpha=1, zorder=10)
        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),modeldata['T2'],c='w',ec=c[ci], alpha=1,label=f'{WRF400_label} {ln} (1 pt)',zorder=10)
        modeldata['T2'] = np.load(f'data/WRF/wrf_temp_evolution/T2_coldlake_{loc}_333m_4pts.npy')
        ax.plot   (modeldata['date']+pd.Timedelta(hours=2),modeldata['T2'],       c=c[ci], alpha=1, zorder=10)
        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),modeldata['T2'],       c='w',ec=c[ci],marker='s', alpha=1,label=f'{WRF400_label} {ln} (4 pts)',zorder=10)
        #ax.plot(modeldata['date']+pd.Timedelta(hours=2),modeldata['T2'],c=c[st],ls='-', lw=lw2*1,alpha=1,label=f'{WRF400_label} {loc} (1 pt)',zorder=-9999)
        #ax.plot(modeldata['date']+pd.Timedelta(hours=2),weighted,       c=c[st],ls='--',lw=lw2*1,alpha=1,label=f'{WRF400_label} {loc} (4 pts)',zorder=-9999)
    
    dates = np.load(f'data/WRF/wrf_profiles-and-timeseries/surface_temp/dates-for-T2.npy')
    markers = ['o','*']
    for fl,tl in enumerate(['cold']):#'warm',
        t_forefield = np.load(f'data/WRF/wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_forefield.npy')[:,0]
        t_valley = np.load(f'data/WRF/wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_valley-TT.npy')[:,0]
        t_valley1 = np.load(f'data/WRF/wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_MG_333m_1pt.npy')
#        ax.scatter(dates[:,0]+pd.Timedelta(hours=2), t_valley-t_forefield, 
#                   c='grey', marker=markers[fl], s=30, zorder=10000)
    #    ax.scatter(dates[:,0], np.load(f'wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_inlet.npy')[:,0], 
    #               vmin=vmin,vmax=vmax, c=np.ones(np.shape(dates[:,0]))*(829), marker=markers[fl], s=300, zorder=10000)
    #    ax.scatter(dates[:,0], np.load(f'wrf_profiles-and-timeseries/surface_temp/T2_{tl}lake_outlet.npy')[:,0], 
    #               vmin=vmin,vmax=vmax, c=np.ones(np.shape(dates[:,0]))*(2700), marker=markers[fl], s=300, zorder=10000)
#        ax.scatter(dates[:,0]+pd.Timedelta(hours=2), t_valley1-t_forefield, 
#                   c='k', marker=markers[fl], s=30, zorder=10000)
#        print (dates[4*24*3+4*16:], np.min((t_valley-t_forefield)[4*24*3+4*16:]), np.min((t_valley1-t_forefield)[4*24*3+4*16:]), (t_valley-t_forefield)[4*24*3+4*16:], (t_valley1-t_forefield)[4*24*3+4*16:])
    
    #for s,loc in zip(range(3),stations):
    ax.set_xlim(start,end)
    ax.set_xlabel('local time')
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(-4.5,13.5)#-18,23)#15)#10,19)#-20,10)#0,30)#-3,20)#
    #ax.legend(loc='best', ncol=6, prop = { "size": 17}) #ncol=4
    ax.grid(True,zorder=-99999)
        
    plt.tight_layout()
#    plt.savefig('plots/aws-wrf_temp.pdf')
    
#    f,ax = plt.subplots(figsize=(15,15))
#    #ax = plotTimeSeries(data=MET_hourly,station_id=MET_hourly['station_id'].unique()[:-2],variable='ws')
#    ax.set_ylabel('wind speed (m/s)')
#    ax.scatter(NB['date'],NB['wspd_u'],s=s,c=c[9],ec='k',lw=lw,label='AWS glacier')
#    #ax.scatter(SM['date'],SM['ws'],s=s,c=c[2],ec='k',lw=lw,label='AWS SM')
#    ax.scatter(FF['date'],FF['WS'],s=s,c=c[1],ec='k',lw=lw,label='AWS inlet')
#    ax.scatter(BH['date'],BH['ws'],s=s,c=c[5],ec='k',lw=lw,label='AWS valley2')
#    #ax.plot(era5['time'],era5['ws'][:,era5_1000_j[0],era5_1000_i[0]],lw=4,alpha=.2,c='k',label=f'ERA5',zorder=-999)# {loc}')
#    for st,loc,ln in zip([1,11,12],[stations[1]]+stations[-3:-1],[locnames[1]]+locnames[-3:-1]):#stations):
#        if st == 0: # valley
#            ci = 2
#        if st == 1: # glacier
#            ci = 9
#        elif st == 11: # valley2
#            ci = 5
#        if st == 12: # inlet
#            ci = 1
#        elif st == 13: # outlet
#            ci = 6
#        lat = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_lat'].iloc[0]
#        lon = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_lon'].iloc[0]
#    #for st,loc in zip(range(3),stations[:3]):#3]):
#    
#        weighted = InverseDistanceWeighted(lat=lat,lon=lon,data=WRF,var='WS')
#    #    np.save(f'WS_{loc}.npy', weighted)
#        i,j = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_i'].unique()[0], WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_j'].unique()[0]
#        modeldata = WRF.loc[(WRF['south_north']==j)&(WRF['west_east']==i),['date','WS']].reset_index(drop=True)
#        ax.plot(modeldata['date']+pd.Timedelta(hours=2),modeldata['WS'],c=c[ci],ls='-', alpha=alp,zorder=10)
#        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),modeldata['WS'],c='w',ec=c[ci],ls='-', alpha=alp,label=f'{WRF1000_label} {ln} (1 pt)',zorder=10)
#    #    ax.plot(modeldata['date']+pd.Timedelta(hours=2),weighted,       c=c[ci],ls='--',alpha=alp,lw=lw2*1,label=f'{WRF1000_label} {loc} (4 pts)',zorder=-9999)
#        
#        weighted = InverseDistanceWeighted(lat=lat,lon=lon,data=WRF400,var='WS')
#        i,j = WRF400_ts.loc[WRF400_ts['station_id']==loc,'grid_i'].unique()[0], WRF400_ts.loc[WRF400_ts['station_id']==loc,'grid_j'].unique()[0]
#        #i,j = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_i'].unique()[0], WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_j'].unique()[0]
#        modeldata = WRF400.loc[(WRF400['south_north']==j)&(WRF400['west_east']==i),['date','WS']].reset_index(drop=True)
#        ax.plot   (modeldata['date']+pd.Timedelta(hours=2),modeldata['WS'],c=c[ci],ls='-', alpha=1,zorder=10)
#        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),modeldata['WS'],c='w',ec=c[ci],ls='-', alpha=1,label=f'{WRF400_label} {ln} (1 pt)',zorder=10)
#        ax.plot   (modeldata['date']+pd.Timedelta(hours=2),weighted,       c=c[ci],ls='-',alpha=1,zorder=10)
#        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),weighted,       c='w',ec=c[ci],ls='-',alpha=1,marker='s',label=f'{WRF400_label} {ln} (4 pts)',zorder=10)
#        
#    ax.set_xlim(start,end)
#    ax.set_ylim(0,12)
#    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
#    handles1, labels1 = ax.get_legend_handles_labels()
#    legend1 = ax.legend(handles1[:3], labels1[:3], ncol=1)
#    #ax.legend(loc=1, ncol=1, prop = { "size": 18})
#    ax.grid(True,zorder=-99999)
#    ax.axvspan(xmin=datetime(2023,9,13), xmax=datetime(2023,9,14), ymin=0, ymax=1, facecolor='grey', alpha=0.1)
#    
#    f,ax = plt.subplots(figsize=(15,15))
#    #ax = plotTimeSeries(data=MET_hourly,station_id=MET_hourly['station_id'].unique()[:-2],variable='wd')
#    ax.set_ylabel('wind direction (\u00b0)')
#    ax.scatter(NB['date'],NB['wdir_u'],s=s,c=c[9],ec='k',lw=lw,label='AWS glacier')
#    #ax.scatter(SM['date'],SM['wd'],s=s,c=c[2],ec='k',lw=lw,label='AWS SM')
#    ax.scatter(FF['date'],FF['WD'],s=s,c=c[1],ec='k',lw=lw,label='AWS inlet')
#    ax.scatter(BH['date'],BH['wd'],s=s,c=c[5],ec='k',lw=lw,label='AWS valley2')
#    #ax.scatter(era5['time'],era5['wd'][:,era5_1000_j[0],era5_1000_i[0]],marker='X',s=2*s,alpha=.2,c='k',label=f'ERA5',zorder=-999)# {loc}')
#    for st,loc,ln in zip([1,11,12],[stations[1]]+stations[-3:-1],[locnames[1]]+locnames[-3:-1]):#stations):
#        if st == 0: # valley
#            ci = 2
#        if st == 1: # glacier
#            ci = 9
#        elif st == 11: # valley2
#            ci = 5
#        if st == 12: # inlet
#            ci = 1
#        elif st == 13: # outlet
#            ci = 6
#    #for st,loc in zip(range(3),stations[:3]):#3]):#1,2 and 1:2
#        #weighted = InverseDistanceWeighted(lat=lat,lon=lon,data=WRF,var='WD')
#        i,j = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_i'].unique()[0], WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_j'].unique()[0]
#        modeldata = WRF.loc[(WRF['south_north']==j)&(WRF['west_east']==i),['date','WD']].reset_index(drop=True)
#        #ax.plot   (modeldata['date']+pd.Timedelta(hours=2),modeldata['WD'],c=c[ci],alpha=alp,zorder=-9999)
#        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),modeldata['WD'],c='none',ec=c[ci],alpha=alp,label=f'{WRF1000_label} {ln} (1 pt)',zorder=10)
#    #    np.save(f'WD_{loc}.npy', modeldata['WD'])
#        #ax.scatter(modeldata['date']+pd.Timedelta(hours=2),weighted,c=c[st],label=f'WRF {loc} (4 pts)',zorder=-9999)
#        
#        #weighted = InverseDistanceWeighted(lat=lat,lon=lon,data=WRF400,var='WD')
#        i,j = WRF400_ts.loc[WRF400_ts['station_id']==loc,'grid_i'].unique()[0], WRF400_ts.loc[WRF400_ts['station_id']==loc,'grid_j'].unique()[0]
#        #i,j = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_i'].unique()[0], WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_j'].unique()[0]
#        modeldata = WRF400.loc[(WRF400['south_north']==j)&(WRF400['west_east']==i),['date','WD']].reset_index(drop=True)
#        #ax.plot   (modeldata['date']+pd.Timedelta(hours=2),modeldata['WD'],c=c[ci],alpha=1,zorder=-9999)
#        ax.scatter(modeldata['date']+pd.Timedelta(hours=2),modeldata['WD'],c='none',ec=c[ci],alpha=1,label=f'{WRF400_label} {ln} (1 pt)',zorder=10)
#        #ax.scatter(modeldata['date']+pd.Timedelta(hours=2),weighted,c=c[st],label=f'WRF {loc} (4 pts)',zorder=-9999)
#    
#    #    if loc == 'NB':
#    ax.set_xlim(start,end)
#    ax.set_yticks(np.arange(0,361,90))
#    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
#    
#    handles1, labels1 = ax.get_legend_handles_labels()
#    legend1 = ax.legend(handles1[:3], labels1[:3], ncol=1)#, columnspacing=0.8, loc='lower left')  
#    #fig.add_artist(legend1)
#    #ax.legend(loc=4, ncol=1, prop = { "size": 18})
#    ax.grid(True,zorder=-99999)
#    ax.axvspan(xmin=datetime(2023,9,13), xmax=datetime(2023,9,14), ymin=0, ymax=1, facecolor='grey', alpha=0.1)
#    
#    plt.show()



# --------------------------------------------------------------------
# plot wrf cross sections
# --------------------------------------------------------------------

xs = np.load('data/WRF/wrf_cross-sections/xs.npy')
ys = np.load('data/WRF/wrf_cross-sections/ys.npy')
ter_line = np.load('data/WRF/wrf_cross-sections/ter_line.npy')
x_labels = np.load('data/WRF/wrf_cross-sections/xlabels.npy', allow_pickle=True)

th_cross_warm = np.load('data/WRF/wrf_cross-sections/th_cross_warmlake_glac2019.npy')
u_cross_warm  = np.load('data/WRF/wrf_cross-sections/u_cross_warmlake_glac2019.npy')
v_cross_warm  = np.load('data/WRF/wrf_cross-sections/v_cross_warmlake_glac2019.npy')
w_cross_warm  = np.load('data/WRF/wrf_cross-sections/w_cross_warmlake_glac2019.npy')
u_tan_warm    = np.load('data/WRF/wrf_cross-sections/u_tan_warmlake_glac2019.npy')

th_cross_cold = np.load('data/WRF/wrf_cross-sections/th_cross_coldlake_glac2019.npy')
u_cross_cold  = np.load('data/WRF/wrf_cross-sections/u_cross_coldlake_glac2019.npy')
v_cross_cold  = np.load('data/WRF/wrf_cross-sections/v_cross_coldlake_glac2019.npy')
w_cross_cold  = np.load('data/WRF/wrf_cross-sections/w_cross_coldlake_glac2019.npy')
u_tan_cold    = np.load('data/WRF/wrf_cross-sections/u_tan_coldlake_glac2019.npy')

th_cross_2006 = np.load('data/WRF/wrf_cross-sections/th_cross_coldlake_glac2006.npy')
u_cross_2006  = np.load('data/WRF/wrf_cross-sections/u_cross_coldlake_glac2006.npy')
v_cross_2006  = np.load('data/WRF/wrf_cross-sections/v_cross_coldlake_glac2006.npy')
w_cross_2006  = np.load('data/WRF/wrf_cross-sections/w_cross_coldlake_glac2006.npy')
u_tan_2006    = np.load('data/WRF/wrf_cross-sections/u_tan_coldlake_glac2006.npy')


wrf_str = []
file_dom = 'wrfout_d01_'
date_for_file = pd.to_datetime("2023-09-12 22:00:00")
for i in range(15):
    wrf_str.append(file_dom+str(date_for_file)[:10]+'_'+str(date_for_file)[11:])
    if date_for_file == pd.to_datetime("2023-09-13 01:00:00") or date_for_file == pd.to_datetime("2023-09-14 01:00:00"):
        date_for_file = date_for_file + pd.Timedelta(hours=4)
    elif date_for_file == pd.to_datetime("2023-09-13 05:00:00") or date_for_file == pd.to_datetime("2023-09-14 05:00:00"):
        date_for_file = date_for_file + pd.Timedelta(hours=2)
    else:
        date_for_file = date_for_file + pd.Timedelta(hours=3)

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)
        

# wind cross section ------------------------------------------------
letters = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)','k)','l)','m)','n)','o)']

def plotWindCrossSection(var='ws'):
    
    plt.rcParams.update({'font.size': 22})
    
    fig, axs = plt.subplots(5,3, sharex=True, sharey=True, figsize=(18,25))
    axs = axs.ravel()
        
    for t in range(15):
        
        thplot = axs[t].contour(xs, ys, th_cross_warm[t], linewidths=1,
                                levels=np.arange(280,320,1), 
                                colors='k')
        axs[t].clabel(thplot,  thplot.levels[::2],
                      inline=1, fmt='%1i', fontsize=10)
        if var == 'ws':
            variable = np.sqrt(u_cross_warm[t]**2+v_cross_warm[t]**2)
            levels=np.arange(0,13)
        elif var == 'w':
            variable = w_cross_warm[t]
            levels=np.arange(-3,3.1,.5)

        tkec = axs[t].contourf(xs, ys, variable,
                               levels=levels,
                               extend='both', 
                               cmap='viridis')#coolwarm')#
        xint = 1; yint = 2; xints = 0; yints = 0
        quiverplot = axs[t].quiver(xs[xints::xint], ys[yints::yint], 
                                   u_tan_warm[t][yints::yint, xints::xint], w_cross_warm[t][yints::yint, xints::xint]*600, 
                                   angles='xy', pivot='mid', scale_units='xy', color='w', width=.005, scale=4, zorder=1000)
        axs[t].quiver(0.6, 310, 10, 0, angles='xy', scale_units='xy', color='w', width=.005, scale=4, zorder=1000)
        axs[t].text(0.6, 110, '10 m/s', color='w', fontsize=14, zorder=1000)
        date = pd.to_datetime(wrf_str[t][-19:-9]+' '+wrf_str[t][-8:])+pd.Timedelta(hours=2)
            
        ht_fill = axs[t].fill_between(xs, 0, (ter_line), facecolor='k', zorder=10)
        axs[t].set_ylim(0, 3000)
        axs[t].plot([1,4.05],[1300,1100], c='w', ls='--', zorder=1000)
        axs[t].plot([8.4,11],[800,620], c='w', ls='--', zorder=1000)
        axs[t].text(4, 780, 'glacier', color='w', rotation=-28, fontsize=14, zorder=1000)
        axs[t].text(19, 150, 'lake', color='w', rotation=0, fontsize=14, zorder=1000)
    
        x_ticks = np.arange(37)#cpairs.shape[0])
        num_ticks = 4
    
        thin = int((len(x_ticks) / num_ticks) + .5)
        axs[t].set_xticks(x_ticks[4::thin])
        axs[t].set_xticklabels(x_labels[4::thin], fontsize=16, rotation=45)
    
        axs[12].set_xlabel(" ")#latitude, longitude")
        axs[13].set_xlabel(" ")#latitude, longitude")
        axs[14].set_xlabel(" ")#latitude, longitude")
        axs[0].set_ylabel("altitude (m a.s.l.)")
        axs[3].set_ylabel("altitude (m a.s.l.)")
        axs[6].set_ylabel("altitude (m a.s.l.)")
        axs[9].set_ylabel("altitude (m a.s.l.)")
        axs[12].set_ylabel("altitude (m a.s.l.)")
        let = letters[t]
        axs[t].set_title(f'{let}      {str(date)[-11:-9]} Sept. {str(date)[-8:-3]} LT        ')#' (local time)')#'{wrf_str[t][11:21]} {wrf_str[t][22:-3]}')  #+2 local time
        
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.18)#, right=0.9, top=0.9)
    cbar_ax = fig.add_axes([0.05, 0.1, 0.9, 0.02])
    
    formatter = LogFormatter(10, labelOnlyBase=False) 
    if var == 'ws':
        label = 'horizontal wind speed (m s$^{-1}$)'
    elif var == 'w':
        label = 'vertical wind speed (m s$^{-1}$)'
    fig.colorbar(tkec, cax=cbar_ax,
                 label=label, orientation='horizontal')
                    
    plt.subplots_adjust(wspace=0.04, hspace=0.2)
    if var == 'ws':
        plt.savefig('plots/wrf_wind.pdf')
    elif var == 'w':
        plt.savefig('plots/wrf_w.pdf')


# sensitivity to lake temperature ------------------------------------------

def plotLakeTempSensitivity(var='ws'):

    # difference plot
    
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(18,5.5))
    axs = axs.ravel()
    plt.rcParams.update({'font.size': 22})
    
    let = ['a)','b)','c)']
    if var == 'ws':
        let = ['d)','e)','f)']
        
    for t,let in zip([0,2,4],let):
        
        if var == 'ws':
            diffvar = np.sqrt(u_cross_cold[t]**2+v_cross_cold[t]**2)-np.sqrt(u_cross_warm[t]**2+v_cross_warm[t]**2)
            vmin = -4; vmax = 4
            levels = np.arange(vmin,vmax+0.1,0.5)
            label='difference in horizontal \n wind speed (m s$^{-1}$)'
        elif var == 'w':
            diffvar = w_cross_cold[t]-w_cross_warm[t]
            vmin = -1.5; vmax = 1.5
            levels = np.arange(vmin,vmax+.01,0.5)
            label='difference in \n vertical velocity (m s$^{-1}$)'
        
        date = pd.to_datetime(wrf_str[t][-19:-9]+' '+wrf_str[t][-8:])+pd.Timedelta(hours=2)
    
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        at = t
        if t == 2:
            at = 1
        if t == 4:
            at = 2
    
        thplot = axs[at].contour(xs, ys, th_cross_cold[t], linewidths=1,
                                levels=np.arange(280,320,1), 
                                colors='k')
        axs[at].clabel(thplot,  thplot.levels[::2],
                      inline=1, fmt='%1i', fontsize=10)
        tkec = axs[at].contourf(xs, ys, diffvar,
                               levels=levels,
                               norm=colors.CenteredNorm(),
                               extend='both',
                               cmap='coolwarm')
        
        xint = 2; yint = 2; xints = 0; yints = 0
        quiverplot = axs[at].quiver(xs[xints::xint], ys[yints::yint], 
                                   u_tan_warm[t][yints::yint, xints::xint], (w_cross_warm[t][yints::yint, xints::xint])*500,#500 for along-valley, 200 for across-valley 
                                   angles='xy', pivot='mid', scale_units='xy', color='grey', width=.005, scale=4, zorder=1000)#5.2
        quiverplot = axs[at].quiver(xs[xints::xint], ys[yints::yint], 
                                   u_tan_cold[t][yints::yint, xints::xint], (w_cross_cold[t][yints::yint, xints::xint])*500,#500 for along-valley, 200 for across-valley 
                                   angles='xy', pivot='mid', scale_units='xy', color='w', width=.005, scale=4, zorder=1000)#5.2
        
        axs[at].quiver(0.6, 360, 10, 0, angles='xy', scale_units='xy', color='w', width=.005, scale=4, zorder=1000)
        axs[at].text(0.6, 110, '10 m/s', color='w', fontsize=14, zorder=1000)
           
        ht_fill = axs[at].fill_between(xs, 0, (ter_line), facecolor='k', zorder=10)
        axs[at].set_ylim(0, 3000)
        axs[at].plot([1,3.95],[1300,1100], c='w', ls='--', zorder=1000)
        axs[at].plot([8.1,11],[820,620], c='w', ls='--', zorder=1000)
        axs[at].text(4.4, 830, 'glacier', color='w', rotation=-22, fontsize=14, zorder=1000)
        axs[at].text(19, 150, 'lake', color='w', rotation=0, fontsize=14, zorder=1000)
    
    
        x_ticks = np.arange(37)#cpairs.shape[0])
        num_ticks = 4
    
        thin = int((len(x_ticks) / num_ticks) + .5)
        axs[at].set_xticks(x_ticks[4::thin])
        axs[at].set_xticklabels(x_labels[4::thin], fontsize=16, rotation=45)
    
        axs[0].set_ylabel("altitude (m a.s.l.)")
        axs[at].set_title(f'{let}                 {str(date)[-11:-9]} Sept. {str(date)[-8:-3]} LT                    ')
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.26)
    cbar_ax = fig.add_axes([1.0, 0.25, 0.01, 0.65])
    
    formatter = LogFormatter(10, labelOnlyBase=False)
    
    cbar = fig.colorbar(tkec, cax=cbar_ax, orientation='vertical', format=tkr.FormatStrFormatter('%.1f'))#, rotation=270)#'horizontal')
    cbar.ax.set_ylabel(label, rotation=270, labelpad=60)
                               
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    
    if var == 'ws':
        plt.savefig('plots/wrf_ws_lake.pdf', bbox_inches="tight")
    elif var == 'w':
        plt.savefig('plots/wrf_w_lake.pdf', bbox_inches="tight")

# sensitivity to glacier extent ------------------------------------------

def plotGlacierSensitivity(var='ws'):

    # difference plot
    
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(18,5.5))
    axs = axs.ravel()
    plt.rcParams.update({'font.size': 22})
    
        
    let = ['a)','b)','c)']
    if var == 'ws':
        let = ['d)','e)','f)']
        
    for t,let in zip([2,4,12],let):

        if var == 'ws':
            diffvar = np.sqrt(u_cross_cold[t]**2+v_cross_cold[t]**2)-np.sqrt(u_cross_2006[t]**2+v_cross_2006[t]**2)
            vmin = -3; vmax = 3
            levels = np.arange(vmin,vmax+0.1,0.5)
            label='difference in horizontal wind speed (m s$^{-1}$)'
        elif var == 'w':
            diffvar = w_cross_cold[t]-w_cross_2006[t]
            vmin = -1.5; vmax = 1.5
            levels = np.arange(vmin,vmax+.01,0.5)
            label='difference in vertical velocity (m s$^{-1}$)'
        
        date = pd.to_datetime(wrf_str[t][-19:-9]+' '+wrf_str[t][-8:])+pd.Timedelta(hours=2)
    
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        at = t
        if t == 2:
            at = 0
        if t == 4:
            at = 1
        if t == 12:
            at = 2
    
        thplot = axs[at].contour(xs, ys, th_cross_cold[t], linewidths=1,
                                levels=np.arange(280,320,1), 
                                colors='k')
        axs[at].clabel(thplot,  thplot.levels[::2],
                      inline=1, fmt='%1i', fontsize=10)
        tkec = axs[at].contourf(xs, ys, diffvar,
                               levels=levels,
                               norm=colors.CenteredNorm(),
                               extend='both',
                               cmap='coolwarm')
        
        xint = 2; yint = 2; xints = 0; yints = 0
        quiverplot = axs[at].quiver(xs[xints::xint], ys[yints::yint], 
                                   u_tan_2006[t][yints::yint, xints::xint], (w_cross_2006[t][yints::yint, xints::xint])*500,#500 for along-valley, 200 for across-valley 
                                   angles='xy', pivot='mid', scale_units='xy', color='grey', width=.005, scale=4, zorder=1000)#5.2
        quiverplot = axs[at].quiver(xs[xints::xint], ys[yints::yint], 
                                   u_tan_cold[t][yints::yint, xints::xint], (w_cross_cold[t][yints::yint, xints::xint])*500,#500 for along-valley, 200 for across-valley 
                                   angles='xy', pivot='mid', scale_units='xy', color='w', width=.005, scale=4, zorder=1000)#5.2
        
        axs[at].quiver(0.6, 360, 10, 0, angles='xy', scale_units='xy', color='w', width=.005, scale=4, zorder=1000)
        axs[at].text(0.6, 110, '10 m/s', color='w', fontsize=14, zorder=1000)
          
        ht_fill = axs[at].fill_between(xs, 0, (ter_line), facecolor='k', zorder=10)
        axs[at].set_ylim(0, 3000)
        axs[at].plot([1,3.95],[1300,1100], c='w', ls='--', zorder=1000)
        axs[at].plot([8.1,11],[820,620], c='w', ls='--', zorder=1000)
        axs[at].text(4.3, 800, 'glacier', color='w', rotation=-22, fontsize=14, zorder=1000)
        axs[at].text(19, 150, 'lake', color='w', rotation=0, fontsize=14, zorder=1000)
    
        x_ticks = np.arange(37)
        num_ticks = 4
    
        thin = int((len(x_ticks) / num_ticks) + .5)
        axs[at].set_xticks(x_ticks[4::thin])
        axs[at].set_xticklabels(x_labels[4::thin], fontsize=16, rotation=45)
    
        axs[0].set_xlabel("latitude, longitude")
        axs[1].set_xlabel("latitude, longitude")
        axs[2].set_xlabel("latitude, longitude")
        axs[0].set_ylabel("altitude (m a.s.l.)")
        #axs[2].set_ylabel("altitude (m a.s.l.)")
        #axs[4].set_ylabel("altitude (m a.s.l.)")
        let = letters[at]
        axs[at].set_title(f'{let}            {str(date)[-11:-9]} Sept. {str(date)[-8:-3]} LT                ')
                    
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    #cbar_ax = fig.add_axes([0.05, 0.1, 0.9, 0.02])
    cbar_ax = fig.add_axes([1.0, 0.12, 0.02, 0.83])
    
    from matplotlib.ticker import LogFormatter 
    formatter = LogFormatter(10, labelOnlyBase=False) 
    cbar = fig.colorbar(tkec, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_ylabel(label, rotation=270, labelpad=25)
                
    plt.tight_layout()

    if var == 'ws':
        plt.savefig('plots/wrf_ws_glacier.pdf', bbox_inches="tight")
    elif var == 'w':
        plt.savefig('plots/wrf_w_glacier.pdf', bbox_inches="tight")
    

# --------------------------------------------------------------------
# plot ERA5 data
# --------------------------------------------------------------------

def plotERA5():
    
    file = f"data/era5_sep2023.nc"
    sld = xr.open_dataset(file)
    
    cmap = pl.cm.YlGn
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.insert(np.linspace(.7, 1, cmap.N-1),0,0)
    my_cmap = ListedColormap(my_cmap)
    
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    lon,lat = np.meshgrid(sld['longitude'],sld['latitude'])
    
    plt.rcParams.update({'font.size': 20})
    
    for t,let in zip([0,1],['a)','b)']):
        projection = ccrs.LambertConformal(central_longitude=10, central_latitude=55)
        crs = ccrs.PlateCarree()
        f = plt.figure(figsize=(16,9), dpi=120)
        ax = plt.axes(projection=projection, frameon=True)
    
        lon_min = -17; lon_max = 27; lat_min = 51; lat_max = 71.5
        
        i = 10; j = 6
    
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.58, "pad" : .03, 'aspect':40, 'label':'wind speed (m s$^{-1}$)'}#total cloud cover'}#Total column water vapour (kg m$^{-2}$)'}#'Snow melt (mm of water equivalent)'}
        cs = ax.contour(lon,lat,(sld["msl"][t]/100).values, transform=ccrs.PlateCarree(), colors='k', levels=np.arange(940, 1060, 2.5), linewidths=1, zorder=100)
        #contour_levels = cs.levels[::2]
        ax.clabel(cs)#,inline=True) #levels=contour_levels,
        (np.sqrt(sld["u10"][t]**2+sld["v10"][t]**2)).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=my_cmap, cbar_kwargs=cbar_kwargs, 
                                                                   levels=np.arange(4,21,2))#(0, 16, 1))#np.arange(0, 51, 5))
        ax.barbs(lon[4::j,4::i],lat[4::j,4::i],sld['u10'][t].values[4::j,4::i]*1.94384,sld['v10'][t].values[4::j,4::i]*1.94384,
                 length=5.5, transform=ccrs.PlateCarree(), color='k', zorder=1000)
    
        ax.add_feature(cf.OCEAN.with_scale("50m"), zorder=0, facecolor='gainsboro')
        ax.add_feature(cf.LAND.with_scale("10m"), zorder=0, facecolor='white',edgecolor='grey')
        ax.add_feature(cf.LAND.with_scale("10m"), zorder=10, facecolor='None', edgecolor='grey')
    
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
        #plt.title(f"{sld['time'].values[t]}")#7. august 2023 kl 12:00")#{sld['time'].values[t]}")#{sl.tcwv[t].time.dt.strftime('%d %B %Y %H:%M').values}")#\n2 m temperature (colours) and snow melt (contours at 1, 2 mm/h)")
        plt.title(f"{let}        {str(sld['time'].values[t])[8:10]+' September 2023, '+str(sld['time'].values[t])[11:16]}         ")
        ax.scatter(7.211611010507808, 61.675952354678884,s=300,marker='X',c=c[1],ec='None',transform=ccrs.PlateCarree(),zorder=2000)
    
        plt.savefig(f"plots/winds_{str(sld['time'].values[t])[:13]}.pdf", dpi=100, format="pdf", bbox_inches='tight')
        plt.show()
