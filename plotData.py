import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from pyproj import Transformer

# CONTENTS

# general plot settings
# plot map
# plot wind
# plot temperature



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
    
    terrain = gpd.read_file('map/Basisdata_4644_Luster_25833_N50Hoyde_GML.gml', layer='Høydekurve')
    lakes = gpd.read_file('map/Basisdata_4644_Luster_25833_N50Arealdekke_GML.gml', layer='Innsjø')
    lakes = lakes.loc[(~np.isnan(lakes['vatnLøpenummer']))&(lakes['høyde']>200)&(lakes['høyde']<400)]
    glaciers = gpd.read_file('map/Basisdata_4644_Luster_25833_N50Arealdekke_GML.gml', layer='SnøIsbre')
    rivers = gpd.read_file('map/Basisdata_4644_Luster_25833_N50Arealdekke_GML.gml', layer='Elv')
    #labels = gpd.read_file('map/Basisdata_4644_Luster_25833_N50Stedsnavn_GML.gml')
    
    terrain_100 = terrain[terrain['høyde'] % 100 == 0]
    terrain_500 = terrain[terrain['høyde'] % 500 == 0]
    glaciers_simp = glaciers.simplify(20, preserve_topology=True)
    glaciers_buffered = glaciers_simp.buffer(.1, resolution=4)
    terrain_glaciers_100 = terrain_100.intersection(glaciers_buffered.unary_union)
    terrain_glaciers_500 = terrain_500.intersection(glaciers_buffered.unary_union)
    
    world = gpd.read_file('map/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
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
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24*.9,13*.9))
    
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
            height = np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/height_{loc}.npy')-100
            mask = (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_13-{h}.npy') > 270) & \
            (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy') > 1)
            if loc == 'FF' or loc == 'NV':
                mask = (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_13-{h}.npy') > 247.5) & \
                (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_13-{h}.npy') < 337.5) & \
                (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy') > 1)
            axes[0,i].plot(np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy'), 
                           height, lw=3, c=colours[j], alpha=alpha)
            axes[0,i].scatter(np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy')[mask], 
                              height[mask], marker='o', lw=1.5, c='none', ec=colours[j])
            axes[0,i].scatter(np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_13-{h}.npy')[~mask], 
                              height[~mask], marker='o', lw=1.5, c='none', ec=colours[j], alpha=alpha)
            if h != '18':
                mask = (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_14-{h}.npy') > 270) & \
                (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy') > 1)
                if loc == 'FF' or loc == 'NV':
                    mask = (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_14-{h}.npy') > 247.5) & \
                    (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/wd_{loc}_14-{h}.npy') < 337.5) & \
                    (np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy') > 1)
                axes[1,i].plot(np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy'), 
                               height, lw=3, c=colours[j], alpha=alpha)
                axes[1,i].scatter(np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy')[mask], 
                                  height[mask], marker='o', lw=1.5, c='none', ec=colours[j])
                axes[1,i].scatter(np.load(f'wrf_profiles-and-timeseries/inverse-distance-weighted/ws_{loc}_14-{h}.npy')[~mask], 
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
                    print ('glac', p, ws, z)
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
        if 270 <= SB.loc[SB['date']==datetime(2023,9,13,h,0,0),'WD'].values[0] <= 360 and SB.loc[SB['date']==datetime(2023,9,13,h,0,0),'WS'].values[0] > 0.5:
            axes[0,i].plot(SB.loc[SB['date']==datetime(2023,9,13,h,0,0),'WS'], 1566, '^', c='k', ms=ms*4)
        else:
            axes[0,i].plot(SB.loc[SB['date']==datetime(2023,9,13,h,0,0),'WS'], 1566, '^', c='k', ms=ms*4, alpha=alpha*3)
        if 270 <= NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wdir_u'].values[0] <= 360 and NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'].values[0] > 0.5:
            #for ax in axes[0, :]:
            #    ax.plot(NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'], 550, 'X', c=c[9], ms=ms*4, alpha=alpha*1.5)
            axes[0,i].plot(NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'], 550, '^', c=c[9], ms=ms*4)
        else:
            #for ax in axes[0, :]:
            #    ax.plot(NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'], 550, 'x', c=c[9], ms=ms*4, alpha=alpha*1.5)
            axes[0,i].plot(NB.loc[NB['date']==datetime(2023,9,13,h-1,30,0),'wspd_u'], 550, '^', c=c[9], ms=ms*4, alpha=alpha*1.5)
        if 247.5 <= FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WD'].values[0] <= 337.5 and FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'].values[0] > 0.5:
            #for ax in axes[0, :]:
            #    ax.plot(FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'], 277, 'X', c=c[1], ms=ms*4, alpha=alpha*1.5)
            axes[0,i].plot(FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'], 277, '^', c=c[1], ms=ms*4)
        else:
            #for ax in axes[0, :]:
            #    ax.plot(FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'], 277, 'x', c=c[1], ms=ms*4, alpha=alpha*1.5)
            axes[0,i].plot(FF.loc[FF['date']==datetime(2023,9,13,h,0,0),'WS'], 277, '^', c=c[1], ms=ms*4, alpha=alpha*1.5)
    for i,h in enumerate([7,12,15]):
        if 270 <= SB.loc[SB['date']==datetime(2023,9,14,h,0,0),'WD'].values[0] <= 360 and SB.loc[SB['date']==datetime(2023,9,14,h,0,0),'WS'].values[0] > 0.5:
            axes[1,i].plot(SB.loc[SB['date']==datetime(2023,9,14,h,0,0),'WS'], 1566, '^', c='k', ms=ms*4)
        else:
            axes[1,i].plot(SB.loc[SB['date']==datetime(2023,9,14,h,0,0),'WS'], 1566, '^', c='k', ms=ms*4, alpha=alpha*3)
        if 270 <= NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wdir_u'].values[0] <= 360 and NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'].values[0] > 0.5:
            #for ax in axes[1, :]:
            #    ax.plot(NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'], 550, 'X', c=c[9], ms=ms*4, alpha=alpha*1.5)
            axes[1,i].plot(NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'], 550, '^', c=c[9], ms=ms*4)
        else:
            #for ax in axes[1, :]:
            #    ax.plot(NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'], 550, 'x', c=c[9], ms=ms*4, alpha=alpha*1.5)
            axes[1,i].plot(NB.loc[NB['date']==datetime(2023,9,14,h-1,30,0),'wspd_u'], 550, '^', c=c[9], ms=ms*4, alpha=alpha*1.5)
        if 247.5 <= FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WD'].values[0] <= 337.5 and FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'].values[0] > 0.5:
            #for ax in axes[1, :]:
            #    ax.plot(FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'], 277, 'X', c=c[1], ms=ms*4, alpha=alpha*1.5)
            axes[1,i].plot(FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'], 277, '^', c=c[1], ms=ms*4)
        else:
            #for ax in axes[1, :]:
            #    ax.plot(FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'], 277, 'x', c=c[1], ms=ms*4, alpha=alpha*1.5)
            axes[1,i].plot(FF.loc[FF['date']==datetime(2023,9,14,h,0,0),'WS'], 277, '^', c=c[1], ms=ms*4, alpha=alpha*1.5)
        
    for ax in axes.flat:
        ax.set_xlim(-1, 11)
        ax.set_ylim(210, 2100)
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

#im = plt.imread('map/2022-08-31-00_00_2022-08-31-23_59_Sentinel-2_L1C_True_color_Jostedalen.png')
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
    
    fig,(ax2,ax) = plt.subplots(2,figsize=(15,17), gridspec_kw={'height_ratios': [1,5]}, dpi=150)
    
    lw = 6
    vmin = -580
    vmax = 4290
    
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
            pl = ax.scatter(ttFF[d],ttFF[t],#.rolling(window=50, center=True).mean(),
                        #ls=ls,
                        c=np.ones(np.shape(ttFF[d]))*(ttFF_hordist[i]),#(np.max(ttFF_elevation)-ttFF_elevation[i])/np.max(ttFF_elevation)),#cm.cool((8-i)/8),
                        s=5,
                        vmin=vmin,vmax=vmax,
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
    cb = plt.colorbar(pl,location='bottom',aspect=60)#,shrink=.5)
    cb.set_label('distance from glacier front (m down-valley)')#elevation (m a.s.l.)')
    plt.tight_layout()
    plt.savefig('plots/temp.pdf', format='pdf')
    
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
# plot ???
# --------------------------------------------------------------------



