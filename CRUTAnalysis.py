import numpy as np
import pandas as pd
import cartopy
import datetime as dtm
import re
import random
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import io
from matplotlib import colors as mcolors
from tqdm import tqdm


class CRUTAnalysis():
    """
       CRUTAnalysis package is a Python framework with useful functions to read, analyze and plot CRUTEM
       data.
       
       Author:
       Giacomo Boldrini 2019 University of Milan-Bicocca
    """

    class Observation():
    
        """
            Observation creates a full model of the observation dataset we are considering.
            Bound methods:
            
            -------------------------------------------------------------------------
            
            __init__:
            self.country : country of the station
            self.lat_line : latitude of the station
            self.lon_line : longitude of the station
            self.id : id of the station
            self.dataset : dataframe with time and temperature (if present)
            self.start_tear : year of the first measure
            self.last_year : year of last measure
            self.interest : If the observation do not contain nans in 1960-1990: True
            else: False
            
            -------------------------------------------------------------------------
            
            temp(self, **kwargs):
            return two lists one with years at monthly resolution and one with
            temperatures (nan are present).
            If no Observations returns empty lists
            
            -------------------------------------------------------------------------
            
            to_dataset(self):
            Returns the dataset of the station. It is accessible through self.dataset
            
            -------------------------------------------------------------------------
            
            annual_nan_mean(self, **kwargs):
            Returns two arrays one with years at annual resolution and one with
            temperatures (nan are present).
            Gives error if station is empty
            
            -------------------------------------------------------------------------
            
            month_res_plotter(self):
            Plots geographical position of the station and its time series (with nan) at
            monthly resolution
            
            -------------------------------------------------------------------------
            
            nan_annual_res_plotter(self):
            Plots geographical position of the station and its time series based on the
            mean without considering nans (still present) at annual resolution
            
            -------------------------------------------------------------------------
        
        """
    
        def __init__(self,file, obs = True):
            
            self.file = file
            
            country, lat_line, lon_line, Number, obs_time_df = self.to_dataset(obs)
            
            self.country = country
            self.lat_line = lat_line
            self.lon_line = - lon_line
            self.id = Number
            self.dataset = obs_time_df
        
        
        def temp(self):
            
            """
                Reads station file and saves time and temperature series. returns empty lists
                if no temperature is found
                
            """
            
            try:
                #this takes care of non utf-8 character in models [1619, 1957, 5387, 6038, 6050]
                data = list(io.open(self.file, errors="replace"))
                skip = data.index("Obs:\n")+1
                data = data[skip:]
                y = [line[:4] for line in data]
                obs = np.array(([re.split(r'\s{1,}', line)[1:13] for line in data])).astype(np.float)
                obs = np.where(obs == -99.0, np.nan, obs)
                
                return y, obs
            
            except:
                return [], []
        
        def to_dataset(self, obs = True):
            
            """
                Generates a dataset of the given station. It loads the country, latitude,
                longitude and id of the given station. Some preventions where made in
                order to open all files even if some have non utf-8 character.
                Arguments:
                obs = True:  If obs is true also temperature are saved, if not temperatures
                are not loaded
                
            """
            
            #this takes care of non utf-8 character in models [1619, 1957, 5387, 6038, 6050]
            data = list(io.open(self.file, errors="replace"))
            
            while True:
                try:
                    country = ([line for line in data if "Country" in line][0].strip('\n').split('=')[1]).strip(' ')
                    break
                except:
                    data.remove([line for line in data if "Country" in line][0])
            
            while True:
                try:
                    lat_line = float(([line for line in data if "Lat" in line][0].strip('\n').split('=')[1]).strip(' '))
                    break
                except:
                    data.remove([line for line in data if "Lat" in line][0])
            
            while True:
                try:
                    lon_line = float(([line for line in data if "Long" in line][0].strip('\n').split('=')[1]).strip(' '))
                    break
                
                except:
                    data.remove([line for line in data if "Long" in line][0])
        
            while True:
                try:
                    Number = ([line for line in data if "Number" in line][0].strip('\n').split('=')[1]).strip(' ')
                    break
                except:
                    data.remove([line for line in data if "Number" in line][0])

            if obs:
                try:
                    #loading temperatures
                    y, obs = self.temp()
                    obs = np.concatenate(obs)
                    
                    #defining first year and last year
                    self.start_year = y[0]
                    self.last_year = y[-1]
            
                    #Defining time values to fill DataFrame:
                    stime = pd.date_range(dtm.datetime.strptime(y[0], "%Y"), dtm.datetime.strptime(str(int(y[-1])+1),"%Y"), freq='M')
                
                    #filling dataframe
                    obs_time_df = pd.DataFrame({'time': stime, Number: obs})
                    obs_time_df = obs_time_df.set_index('time')
                
                except:
                    print("No observation on {} in {}".format(Number, country))
                    obs_time_df = pd.DataFrame()
                
                return country, lat_line, lon_line, Number, obs_time_df
            
            else:
                return country, lat_line, lon_line, Number, _

        def nan_interest_period(self, **kwargs):
            """
                For the given station it checks if temperature are present in the selected time
                period, appends the result in the object information as a boolean
            """
            try:
                time = self.dataset.index
                start = np.where(time == '1960-01-31')[0][0]
                stop = np.where(time == '1990-01-31')[0][0]
                obs = self.dataset[start:stop]
                if any(np.isnan(obs.values)) == False:
                    self.interest = True
                else:
                    self.interest = False
            except:
                self.interest = False
        
        
        def annual_nan_mean(self, **kwargs):
            """
                Computes annual mean ignoring nans for the given station and returns it
            """
            y, obs = self.temp()
            annual_mean = np.nanmean(obs , axis = 1)
            annual_std = np.nanstd(obs , axis = 1)
            year = pd.date_range(dtm.datetime.strptime(y[0], "%Y"), dtm.datetime.strptime(str(int(y[-1])+1),"%Y"), freq='Y')
            
            return year, annual_mean, annual_std
        
        def monthly_nan_mean(self, **kwargs):
            """
                Computes monthly mean for each year for the given stations
                returns: months, monthly_mean, monthly_std
            """
            
            y, obs = self.temp()
            monthly_mean = np.nanmean(obs , axis = 0)
            monthly_std = np.nanstd(obs , axis = 0)
            months = ['gen', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            return months, monthly_mean, monthly_std
        
        def nan_month_res_plotter(self, out_path=None, Save = False):
            """
                Plotting Monthly time series for the given station.
                Arguments:
                out_path = None: Out_path given by user for saving the plot, if save is True
                Save = False: if True Save the plot
            """
            
            fig = plt.figure(figsize=(13,4))
            fig.suptitle("Monthly resolution temperature of {}: {},{} station {}".format(self.country, self.lat_line,self.lon_line, self.id))
            
            ax = fig.add_subplot(1,2,1,projection=ccrs.PlateCarree())
            ax.set_global()
            ax.stock_img()
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            ax.add_feature(cfeature.BORDERS, edgecolor='grey')
            ax.add_feature(cfeature.LAND)
            ax.plot([self.lon_line], [self.lat_line], color='red', linewidth=10, marker='o', markersize=3, transform=ccrs.Geodetic())
            
            ax1 = fig.add_subplot(1,2,2)
            ax1.plot(self.dataset, c='orange')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Month temperature [°C]')
            
            if Save:
                fig.savefig(out_path + '/nan_month_res_{}.pdf'.format(self.id))
                plt.close()



        def nan_annual_res_plotter(self, out_path=None, Save = False):
            """
                Plotting time series at annual resolution for the given station.
                Arguments:
                out_path = None: Out_path given by user for saving the plot, if save is True
                Save = False: if True Save the plot
            """
            
            year, annual_mean, annual_std = self.annual_nan_mean()
            fig = plt.figure(figsize=(13,4))
            fig.suptitle("Nan annual resolution temperature of {}: {},{} station {}".format(self.country, self.lat_line,self.lon_line, self.id))
            
            ax = fig.add_subplot(1,2,1,projection=ccrs.PlateCarree())
            ax.set_global()
            ax.stock_img()
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            ax.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax.add_feature(cfeature.LAND)
            ax.plot([self.lon_line], [self.lat_line], color='red', linewidth=10, marker='o', markersize=3, transform=ccrs.Geodetic())
            
            ax1 = fig.add_subplot(1,2,2)
            ax1.errorbar(year, annual_mean, yerr = annual_std, marker = 'o', linestyle = '--', c='orange')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Year temperature [°C]')
            
            if Save:
                fig.savefig(out_path + '/nan_annual_res_{}.pdf'.format(self.id))
                plt.close()
            
            return year, annual_mean

        def mean_temp_monthly(self, out_path=None, Save = False):
            """
                Plotting monthly mean for each year for the given station.
                Arguments:
                out_path = None: Out_path given by user for saving the plot, if save is True
                Save = False: if True Save the plot
            """
            month, monthly_mean, monthly_std = self.monthly_nan_mean()
            
            fig = plt.figure(figsize=(13,4))
            fig.suptitle("Nan monthly mean temperature of {}: {},{} station {}".format(self.country, self.lat_line,self.lon_line, self.id))
            
            ax = fig.add_subplot(1,2,1,projection=ccrs.PlateCarree())
            ax.set_global()
            ax.stock_img()
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            ax.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax.add_feature(cfeature.LAND)
            ax.plot([self.lon_line], [self.lat_line], color='red', linewidth=10, marker='o', markersize=3, transform=ccrs.Geodetic())
            
            ax1 = fig.add_subplot(1,2,2)
            ax1.errorbar(month, monthly_mean, yerr = monthly_std, marker = 'o', linestyle = '--', c='orange')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Mean temperature [°C]')
            
            if Save:
                fig.savefig(out_path + '/mean_temp_monthly_{}.pdf'.format(self.id))
                plt.close()
            
            return month, monthly_mean

        def monthly_plus_annual(self, out_path=None, Save = False):
            """
                Plotting both annual and monthly time series for the given station.
                Arguments:
                out_path = None: Out_path given by user for saving the plot, if save is True
                Save = False: if True Save the plot
            """
            year, annual_mean, _ = self.annual_nan_mean()
            
            fig = plt.figure(figsize=(13,4))
            fig.suptitle('Annual mean of station {} observation/ {}'.format(self.id, self.country))
            
            ax = fig.add_subplot(1,2,1,projection=ccrs.PlateCarree())
            ax.set_global()
            ax.stock_img()
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            ax.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax.add_feature(cfeature.LAND)
            ax.plot([self.lon_line], [self.lat_line], color='red', linewidth=10, marker='o', markersize=3, transform=ccrs.Geodetic())
            
            ax1 = fig.add_subplot(1,2,2)
            ax1.plot(self.dataset, c = 'orange', label = 'monthly time series', zorder=1)
            ax1.plot(year, annual_mean, c = 'red', label = 'annual mean series',zorder=10)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Temperature [°C]')
            ax1.legend(loc = 'lower right')
            
            
            if Save:
                fig.savefig(out_path + '/monrtly_plus_annual_{}.pdf'.format(self.id))
                plt.close()
            
            return year, annual_mean

        def monthly_reconstruction(self, out_path=None, Save = False):
            """
                Reconstructing monthly time series with the monthly mean for missing observations
                and plotting it.
                Arguments:
                out_path = None: Out_path given by user for saving the plot, if save is True
                Save = True: if True Save the plot
            """
            _, monthly_mean, _ = self.monthly_nan_mean()
            time = self.dataset.index
            recon = []
            for i, j in zip(self.dataset.values, time):
                
                if np.isnan(i):
                    month = j.to_pydatetime().month
                    recon.append(monthly_mean[month-1])
                
                else:
                    recon.append(i)
            
            fig = plt.figure(figsize=(13,4))
            fig.suptitle('Reconstruction missing month with mean {}/{}'.format(self.id, self.country))
            
            ax = fig.add_subplot(1,2,1,projection=ccrs.PlateCarree())
            ax.set_global()
            ax.stock_img()
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            ax.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax.add_feature(cfeature.LAND)
            ax.plot([self.lon_line], [self.lat_line], color='red', linewidth=10, marker='o', markersize=3, transform=ccrs.Geodetic())
            
            ax1 = fig.add_subplot(1,2,2)
            ax1.plot(self.dataset, c = 'plum', label = 'monthly time series', zorder=10)
            ax1.plot(time, recon, c= 'orange', label = 'reconstruction with mean', zorder=2)
            ax1.legend(loc = 'lower right')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Temperature [°C]')
            
                                            
            if Save:
                fig.savefig(out_path + '/montly_reconstruction_{}.pdf'.format(self.id))
                plt.close()
            
            return time, recon

                
                
    class Full_stations():
        """
            Full_stations work with the whole dataset, creating dataset, plotting, and gridding over
            geographical position
            
        """
        
        def __init__(self, files):
            self.file_list = files

        def to_dataset(self, num_stations = 1000, Random = False):
            
            """
                Taking the file list and fills the dataset with time as index and as
                columns the id of the stations. Info are accessible from self.df or
                as self.metadata respectively.
                
                Arguments:
                num_stations = 1000, selects the number of stations to construct the
                dataset.
                Random = False, choose to select randomly the stations to have equally
                distributed stations in the analysis.
            """
            #generate date_range and initialize dataframe
            taxis = pd.date_range('1850-01', '2019-01', freq='M')
            df = pd.DataFrame({'time': taxis})
            
            #setting date_range as the index of the dataset
            df = df.set_index(['time'])
            
            #defining dataset with metadata:
            lat = []
            lon = []
            count = []
            ids = []
            
            #counter for visualization of progress
            counter = 0
            
            #extracting random stations if random = True is passed when calling method
            if Random:
                file_list = random.sample(self.file_list, num_stations)
                
                #redifine the file_list
                self.file_list = file_list
            
            else:
                file_list = self.file_list
        
            for file in tqdm(file_list):
                
                #from class Observation retrieve years and observations
                station = CRUTAnalysis.Observation(file)
                
                try:
                    #merging with the final dataframe on column time since it is common
                    df = pd.merge(df, station.dataset, on = 'time', how='left')
                    #filling metadata lat, lon, country and id of station
                    lat.append(station.lat_line)
                    lon.append(station.lon_line)
                    count.append(station.country)
                    ids.append(station.id)
                
                except:
                    print('Cannot merge station {}/{}, observation = {}'. format(station.id, station.country, len(station.dataset)))
                    self.file_list.remove(file)
                    continue
            
            #inserting df in the model variables
            self.df = df
            #print(len(ids), len(lon), len(lat), len(count), len(self.file_list))
            #inserting metadata in a datagrame and setting ids as index
            metadata = pd.DataFrame({'ids': ids, 'lon': lon, 'lat': lat, 'country': count})
            metadata = metadata.set_index('ids')
            
            #inserting metadata in the model variables
            self.metadata = metadata
            self.ids_list = ids
            
            #define recover attributes
            
            self.recover = self.df
            self.meta_recover = self.metadata

        def save_as_attr(self, **kwargs):
            """
                Saving user passed arguments as class atrributes
                Arguments:
                **kwargs: keyworded arguments the name will be the name of the attribute, save_as_attr(name=att)
                will save it in self as self.name = att
            """
            for key, value in kwargs.items():
                setattr(self, key, value)


        def recover(self):
            """
                Recover original object attributes.
            """
        
            #recovery
            self.df = self.recover
            self.metadata = self.meta_recover
        
            #deleting everything else to clean memory
            if self.ext_lat:
                del(self.ext_lat)
            if self.ext_lon:
                del(self.ext_lon)
            if self.grid:
                del(self.grid)
            if self.ids_list:
                del(self.ids_list)
            if self.meta_slice:
                del(self.meta_slice)
            if self.sliced:
                del(self.sliced)

        def plot_all(self, out_path=None, Save = False, num_stations=1000):
                        
            """
                Plotting all stations on a map, divided in color of stations with
                full observations in the reference period and not.
                
                Arguments:
                num_stations = 1000, selects the number of stations to construct the
                dataset.
                out_path, Path of the folder to save pdf of plots
                Save = False, choosing to hide plots and not saving them.
            """
            #plotting stations with full data in period 1960-1990:
            fig = plt.figure(figsize=(12,6))
            fig.suptitle("Stations with (red) and without (green) full interset period")

            ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
            ax.set_global()
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            ax.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax.add_feature(cfeature.LAND)

            w_interest_lon = []
            w_interest_lat = []
            wo_interest_lon =[]
            wo_interest_lat =[]
                                    
            for i in tqdm(range(num_stations)):
                
                try:
                    Obs = CRUTAnalysis.Observation(self.file_list[i])
                    Obs.nan_interest_period()
                    if Obs.interest:
                        w_interest_lon.append([Obs.lon_line])
                        w_interest_lat.append([Obs.lat_line])
                        ax.plot([Obs.lon_line], [Obs.lat_line], color='red', linewidth=1, marker='o', markersize=2, transform=ccrs.Geodetic())
                    else:
                        wo_interest_lon.append([Obs.lon_line])
                        wo_interest_lat.append([Obs.lat_line])
                        ax.plot([Obs.lon_line], [Obs.lat_line], color='green', linewidth=1, marker='o', markersize=2, transform=ccrs.Geodetic())
                except:
                    continue
                                                                        
                                                                        
            #plotting separately stations with interest period and without interest period
            fig1 = plt.figure(figsize=(12,6))
            fig1.suptitle("Stations with full interset period")
        
            ax1 = fig1.add_subplot(1,1,1,projection=ccrs.PlateCarree())
            ax1.set_global()
            ax1.coastlines()
            ax1.gridlines(draw_labels=True)
            ax1.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax1.add_feature(cfeature.LAND)
            plt.scatter(w_interest_lon, w_interest_lat, color='red',s=3, linewidth=1, marker='o', transform=ccrs.Geodetic())
            
            fig2 = plt.figure(figsize=(12,6))
            fig2.suptitle("Stations without full interset period")
                
            ax2 = fig2.add_subplot(1,1,1,projection=ccrs.PlateCarree())
            ax2.set_global()
            ax2.coastlines()
            ax2.gridlines(draw_labels=True)
            ax2.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax2.add_feature(cfeature.LAND)
            plt.scatter(wo_interest_lon, wo_interest_lat, color='green',s = 3, linewidth=1, marker='o', transform=ccrs.Geodetic())
                    
                    
            if Save:
                #Saving plot
                fig.savefig(out_path + '/All_Stations_Crutem.pdf')
                fig1.savefig(out_path + '/Stations_full_interest.pdf')
                fig2.savefig(out_path + '/Station_wo_full_interest.pdf')
                plt.close()
                    
                    
        def regridding(self, lat_ext = [-90, 90], lon_ext = [-180, 180], cell_step = 5, sliced = False):
            """
                Regridding the dataset in given cell steps. Returns the gridded dataframe
                with the latitude ranges and station indices, will be accessible through
                self.dataframe .
                
                Arguments:
                cell_step = 5, size of the square cell. If the number is not a common divisor
                of 360 and 180 the functions return and gives error
            """
                                
            if sliced:
                
                #flooring and ceiling if sliced is called, to first integer divisible for the cell_step
                lat_ext = [int(cell_step * mt.floor(self.ext_lat[0]/cell_step)), int(cell_step * mt.ceil(self.ext_lat[1]/cell_step))]
                lon_ext = [int(cell_step * mt.floor(self.ext_lon[0]/cell_step)), int(cell_step * mt.ceil(self.ext_lon[1]/cell_step))]
                    
            else:
                if (sum(lat_ext))%cell_step != 0 or (sum(lon_ext))%cell_step != 0:
                    print('cell_step must be a common divisor of 360 and 180')
                    return
                                
            grid_dataframe = pd.DataFrame(columns = ['min_lat', 'max_lat', 'min_long', 'max_long', 'indices','number_station'])

            #defining latitudes and longitudes values
            lat = np.arange(lat_ext[0],lat_ext[1]+cell_step, cell_step)
            long = np.arange(lon_ext[0], lon_ext[1]+cell_step, cell_step)
            
            dataframe_count = 0
                    
                    
            #cycling to fill the dataframe of stations and temps
            for i in tqdm(range(lat.shape[0]-1)):
                        
                for j in range(long.shape[0]-1):
                            
                    #defining stations in the grid
                    if sliced:
                        loc = self.meta_slice.loc[(self.meta_slice['lat']>=lat[i]) & (self.meta_slice['lat']<lat[i+1]) & (self.meta_slice['lon']>= long[j]) & (self.meta_slice['lon']<long[j+1])]
                    else:
                        loc = self.metadata.loc[(self.metadata['lat']>=lat[i]) & (self.metadata['lat']<lat[i+1]) & (self.metadata['lon']>= long[j]) & (self.metadata['lon']<long[j+1])]
                    #checking for stations and filling in some way the dataframe
                    if loc.empty == False:
                        indices = []
                        for ind in loc.index:
                            if sliced:
                                if ind in self.sliced.columns:
                                    indices.append(ind)
                            else:
                                if ind in self.df.columns:
                                    indices.append(ind)
                                                                            
                                                                            
                        if len(indices) == 0:
                            grid_dataframe.loc[dataframe_count] = [lat[i], lat[i+1], long[j], long[j+1], 0, 0]
                        else:
                            grid_dataframe.loc[dataframe_count] = [lat[i], lat[i+1], long[j], long[j+1], indices, len(indices)]
                    else:
                        grid_dataframe.loc[dataframe_count] = [lat[i], lat[i+1], long[j], long[j+1], 0,0]
                    dataframe_count += 1
                                                    
            self.grid = grid_dataframe
                                                        
            return grid_dataframe

        def cleansing(self, out_path=None, Save = False, sliced = False):
            """
            cleaning the dataframe of stations keeping only the stations with a
            complete reference period 1960-1990. Subsequently it plots the result and
            updated self.metadata and self.df
            
            Arguments:
            out_path, Path to save plots.
            Save = False, Tell if the plots are going to be saved.
            verbosity = False, Show or not show plots of the cleaning.
            """
            
            taxis = pd.date_range('1850-01', '2019-01', freq='M')
            dropped = pd.DataFrame({'time': taxis})
            dropped = dropped.set_index(['time'])
        
            try:
                if sliced:
                    self.sliced

                else:
                    self.df
        
            except:
                print('Full_dataset object needs to be converted into dataframe before cleansing. You can type data = Full_stations(file_list).to_dataset()')
                return


            if sliced:
                for i in tqdm(self.meta_slice.index, desc = 'Cleaning df'):
                    if any(np.isnan(self.sliced[i]['1960-01-31':'1990-01-31'])):
                        dropped = pd.merge(dropped, self.sliced[i], on='time', how = 'left')
                        self.sliced = self.sliced.drop(i, 1)
                        self.meta_slice = self.meta_slice.drop(index = i, axis = 0)
                        self.df = self.df.drop(i, 1)
                        self.metadata = self.metadata.drop(index = i, axis = 0)

            else:
                for i in tqdm(self.df.columns, desc = 'Cleaning df'):
                    if any(np.isnan(self.df[i]['1960-01-31':'1990-01-31'])):
                        dropped = pd.merge(dropped, self.df[i], on='time', how = 'left')
                        self.df = self.df.drop(i, 1)
                        self.metadata = self.metadata.drop(index = i, axis = 0)
                    
            #Cleaning the gridded dataframe from dropped stations
        
            try:
                #if grid exist we go on
                self.grid
                #retrieve indices to apply loc.
                df_indices = self.grid[self.grid['indices'] != 0].index
                for i in tqdm(df_indices, desc = 'Grid cleaning'):
                    #redefine station_id indices of gridded dataset removing the ones dropped
                    self.grid.loc[i].indices = [x for x in self.grid.loc[i].indices if x not in dropped.columns]
                    self.grid.loc[i].number_station = len(self.grid.loc[i].indices)
                    #if the list is empty after the cleansing then we set the entry to zero
                    if len(self.grid.loc[i].indices) == 0:
                        self.grid.loc[i].indices = 0
                        self.grid.loc[i].number_station = 0
            except:
                pass
        
            if Save:
                fig = plt.figure(figsize=(12,6))
                fig.suptitle("Visualizing cleansing")
                ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
                ax.gridlines(draw_labels=True)
                ax.coastlines()
                ax.stock_img()
                ax.add_feature(cfeature.LAND)
                ax.set_global()
                ax.add_feature(cfeature.BORDERS, edgecolor='gray')
            
            if sliced:
                ax.set_extent([self.ext_lon[0]-3, self.ext_lon[1]+3, self.ext_lat[0]-3, self.ext_lat[1]+3], ccrs.PlateCarree())
                ax.plot(self.meta_slice['lon'].values, self.meta_slice['lat'].values, 'bo', markersize=2, transform=ccrs.Geodetic())
            
            else:
                count = 0
                while count < len(self.metadata):
                    ax.plot(self.metadata.lon.values[count], self.metadata.lat.values[count], color='blue', marker='o', markersize=2, transform=ccrs.Geodetic())
                    count += 1

            fig.savefig(out_path + "/Cleansing_slicing:{}.pdf".format(sliced))

        def grid_contribution(self, grid, out_path=None, Save = False, sliced = False, verbosity = True):
                """
                    Plotting which grids contribute with a station marked with an X.
                    
                    Arguments:
                    grid, Takes the grid output of self.regridding to plot
                    out_path, Path to save plots.
                    Save = False, Tell if the plots are going to be saved.
                    verbosity = False, Show or not show plots of the cleaning.
                    
                """
                #plotting contribution cells along with point of the stations
                #it is defined as a self method of Full_Stations to be able to
                #record all the files simply. It still takes as input the grid which
                #needs to be defined outside
                
                #getting the steps of the grid
                grid_lon_step = grid.loc[0, :]['max_long'] - grid.loc[0, :]['min_long']
                grid_lat_step = grid.loc[0, :]['max_lat'] - grid.loc[0, :]['min_lat']

                #plotting the grid based on the grid step
                fig = plt.figure(figsize=(12,8))
                ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
                ax.set_global()
                ax.add_feature(cfeature.LAND)
                if sliced:
                    lat_ext = [int(grid_lat_step* mt.floor(self.ext_lat[0]/grid_lat_step)), int(grid_lat_step * mt.ceil(self.ext_lat[1]/grid_lat_step))]
                    lon_ext = [int(grid_lon_step * mt.floor(self.ext_lon[0]/grid_lon_step)), int(grid_lon_step * mt.ceil(self.ext_lon[1]/grid_lon_step))]
                
                #ax.gridlines(xlocs=range(mt.floor(self.ext_lon[0]),mt.floor(self.ext_lon[1]),grid_lon_step), ylocs=range(mt.floor(self.ext_lat[0]),mt.floor(self.ext_lat[1]+1),grid_lat_step))
                    #ax.set_extent([mt.floor(self.ext_lon[0]),mt.ceil(self.ext_lon[1]), mt.floor(self.ext_lat[0]),mt.ceil(self.ext_lat[1])], ccrs.PlateCarree())
                    ax.gridlines(xlocs=range(lon_ext[0],lon_ext[1]+grid_lon_step,grid_lon_step), ylocs=range(lat_ext[0], lat_ext[1]+grid_lat_step,grid_lat_step))
                    ax.set_extent([lon_ext[0], lon_ext[1], lat_ext[0], lat_ext[1]], ccrs.PlateCarree())
                else:
                    ax.gridlines(xlocs=range(-180,181,grid_lon_step), ylocs=range(-90,91,grid_lat_step))
                                                    
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, edgecolor='gray')
                                                
                #cycling on all grid entries and plotting
                for i in tqdm(range(len(grid))):
                    
                    #if the indices of the grid is not null (so the cell contains stations)
                    if grid.loc[i, :]['indices'] != 0:
                                                            
                            #plots an x on the central latitude and longitude
                            central_lat = grid.loc[i, :]['max_lat'] - (grid.loc[i, :]['max_lat'] - grid.loc[i, :]['min_lat'])/2.
                            central_lon = grid.loc[i, :]['max_long'] - (grid.loc[i, :]['max_long'] - grid.loc[i, :]['min_long'])/2.
                            ax.plot([central_lon], [central_lat], color='red', marker='x', markersize=7, transform=ccrs.Geodetic())
                                                        
                            for i in grid.loc[i, :]['indices']:
                                #for each station in the cell plots a point at its coordinate. Coordinates
                                #retrieved from the metadata dataset from self
                                if sliced:
                                    ax.plot(self.meta_slice.loc[i]['lon'], self.meta_slice.loc[i]['lat'], color='blue', marker='o', markersize=1, transform=ccrs.Geodetic())
                                else:
                                    ax.plot(self.metadata.loc[i]['lon'], self.metadata.loc[i]['lat'], color='blue', marker='o', markersize=1, transform=ccrs.Geodetic())
                                                                                
                if not verbosity:
                    plt.close()
                                                                                        
                if Save:
                    fig.savefig(out_path + '/Contributing_Areas.pdf')
                    plt.close()

        def grid_series_plotter(self, grid, out_path=None, how_many = 1, Save = False, anomaly = False,  random = True, complete_months = True, Monthly = True, incomplete_months = True):
            """
                Choosing random grids and plots subsequentially all the monthly time
                series, annual average with incomplete years and with only complete years.
                
                Arguments:
                grid, Takes the grid output of self.regridding to plot
                out_path, Path to save plots.
                how_many = 1, Telling function how many cells we want to plot
                random = False, Choose if random or linear in file list order
                Save = False, Tell if the plots are going to be saved.
                complete_months = True, Choosing to plot the series for only years with
                complete months.
                Monthly = True, Choosing to plot all monthly time series for stations in
                the cell.
                incompolete_months = True, Choosing to plot annual time series of
                years with incomplete months
            """
            #finding only cells with stations in the grid and define a new dataset:
            filled_cells = grid[grid['indices'] != 0].copy().reset_index()
        
            #if user want all the filled cells we set how_many = number cells
            if how_many == all:
                how_many = len(filled_cells)
            
            #if not how_many==all then we check if how many is more than the number of filled cells
            if how_many > len(filled_cells):
                print('how_many value is > than the number of accessible filled cells which is: {}'.format(len(filled_cells)))
                return
        
            #choosing random cells to plot time series. Default=1 cell ( how_many )
            drop_cells = np.random.choice(len(filled_cells), len(filled_cells)-how_many, replace=False)
            filled_cells.drop(drop_cells, inplace= True)
            
            fig = plt.figure(figsize=(15,5*how_many))
            fig.suptitle('Time series of grid dataframe')
            count_axes = 1
                
            for i in filled_cells['indices']:
                ax = fig.add_subplot(len(filled_cells), 1, count_axes)
                
                #defining temporal dataframe to analyze cells
                taxis = pd.date_range('1850-01', '2019-01', freq='M')
                cell_month_df = pd.DataFrame({'time': taxis})
                cell_month_df = cell_month_df.set_index(['time'])
                
                
                for j in i:
                    
                    if anomaly:
                        cell_month_df = pd.merge(cell_month_df, self.df[j]-self.df[j][anomaly[0]:anomaly[1]].mean(skipna =True), on='time', how='left')
                    else:
                        cell_month_df = pd.merge(cell_month_df, self.df[j], on='time', how='left')
                
                                                    
                    #plotting annual mean with incomplete years
                    cell_annual_inc = cell_month_df.copy()
                    cell_annual_inc = cell_annual_inc.groupby(pd.Grouper(freq="Y")).mean()
                    cell_annual_inc = cell_annual_inc.mean(axis = 1)
                    
                    #deleting incomplete years:
                    cell_annual = cell_month_df.copy()
                    
                    count = 0
                    years_to_drop = []
                    
                    #filling a vector with the years corresponding to empty months
                    while count < len(cell_annual):
                        if all(np.isnan(cell_annual.iloc[count])) == True:
                            years_to_drop.append(cell_annual.index[count].year)
                        count += 1
                
                
                    #taking just single values for each year
                    dropping = np.unique(years_to_drop)

                    #saving old index years-month-day
                    vecchio_indice = cell_annual.index
                    
                    #reindexing only on year
                    cell_annual.index = cell_annual.index.year
                    
                    #dropping rows if index = year in dropping list
                    df1 = cell_annual.drop(index = dropping, axis = 0)
                    
                    nuovo_indice = [i for i in vecchio_indice if i.year not in dropping]
                    df1 = df1.set_index([nuovo_indice])
                    
                    #averaging over year and over all cells
                    df1 = df1.groupby(pd.Grouper(freq="Y")).mean()
                    df1 = df1.mean(axis = 1)
            
                #computing monthly mean of cell:
                cell_month_mean = cell_month_df.mean(axis = 1)
                
                ax.set_title('Stations: {}'.format(len(i)))
                if Monthly:
                    plt.plot(cell_month_df, c= 'lightskyblue', linewidth = .5)
                    plt.plot(cell_month_mean, c = 'blue', linestyle = ':', linewidth = 1)

                if incomplete_months:
                    plt.plot(cell_annual_inc, c = 'gold', linewidth = 2, label = 'Annual mean inc')

                if complete_months:
                    plt.plot(df1, c = 'red', linewidth = 2, label = 'Annual mean only complete')

                plt.ylabel('Temperature [°C]')
                plt.legend(loc = 'lower right')
                count_axes += 1
                
                if Save:
                    #fig.tight_layout()
                    fig.savefig(out_path + '/grid_series.pdf')
                    plt.close()


            def slice_region(self, min_lat = None, max_lat = None, min_lon = None, max_lon = None, out_path = False, region = False, Save = False):
                                
                """
                    Slicing on a regiorn of interest. Returns the dataframe for the given region.
                    Pay attention to slice before or after cleansing the data.
                    Arguments:
                    min_lat, minimum latitude of the region.
                    max_lat, maximum latitude of the region.
                    min_lon, minimum longitude of the region.
                    max_lon, maximum longitude of the region.
                    out_path, Path where the plots are saved.
                    Save = False, choose to save the plots of the region or not
                """
                                        
                if region and not all([min_lat, max_lat, min_lon, max_lon]) :
                                            
                    self.meta_slice = self.metadata[self.metadata['country'].str.contains(region)]
                    self.sliced = self.df[self.meta_slice.index]

                    lat = self.meta_slice.lat.values
                    lon = self.meta_slice.lon.values
                    self.ext_lat = [min(self.meta_slice.lat.values), max(self.meta_slice.lat.values)]
                    self.ext_lon = [min(self.meta_slice.lon.values), max(self.meta_slice.lon.values)]
                    #for plotting
                    min_lon = self.ext_lon[0]
                    max_lon = self.ext_lon[1]
                    min_lat = self.ext_lat[0]
                    max_lat = self.ext_lat[1]
                                                            
                else:
                                                                
                    self.meta_slice = self.metadata.loc[(self.metadata['lat']<= max_lat) & (self.metadata['lat']>=min_lat) & (self.metadata['lon']<=max_lon) & (self.metadata['lon']>=min_lon)]
                    self.sliced = self.df[self.meta_slice.index]
                    lat = self.metadata.lat.values
                    lon = self.metadata.lon.values
                    self.ext_lat = [min_lat, max_lat]
                    self.ext_lon = [min_lon, max_lon]
                                                                            
                                                                            
                if Save:
                                                                                
                    fig = plt.figure(figsize=(10,5))
                    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
                    ax.gridlines(draw_labels=True)
                    ax.coastlines()
                    ax.stock_img()
                    ax.add_feature(cfeature.LAND)
                    ax.set_global()
                    ax.set_extent([self.ext_lon[0]-3, self.ext_lon[1]+3, self.ext_lat[0]-3, self.ext_lat[1]+3], ccrs.PlateCarree())
                    ax.add_feature(cfeature.BORDERS, edgecolor='gray')
                    ax.plot(lon, lat, 'bo', markersize=2, transform=ccrs.Geodetic())
                    fig.savefig(out_path + '/slice.pdf')

            def Grid_mesh(self, monthly = False, sliced = False, meshed = False):
    
                #computing central latitudes and central longitudes
                grid_lon_step = self.grid.loc[0, :]['max_long'] - self.grid.loc[0, :]['min_long']
                grid_lat_step = self.grid.loc[0, :]['max_lat'] - self.grid.loc[0, :]['min_lat']
                
                #defining axis for annual mean of the cell. Since it will be an int64 array we can't define a data_range
                taxis = np.arange(1850, 2020, 1)
                #defining axis for monthly measures of the stations
                taxis_month = pd.date_range('1850-01', '2019-01', freq='M')
                
                #dataframe containing all the annual time series for the cells
                all_filled_cells_annual_mean = pd.DataFrame({'time': taxis})
                all_filled_cells_annual_mean = all_filled_cells_annual_mean.set_index(['time'])
                
                if monthly:
                    all_filled_cells_monthly = pd.DataFrame({'time': taxis_month})
                    all_filled_cells_monthly = all_filled_cells_monthly.set_index(['time'])
            
                #array with central latitudes, central longitudes and cell annual mean
                total_central_lat = []
                total_central_lon = []
                total_cell_annual_mean = []
                
                #cycling on all the cells
                for i in tqdm(range(len(self.grid))):
                    
                    #computing central lat and central lon
                    central_lat = self.grid.loc[i]['max_lat'] - (self.grid.loc[i]['max_lat'] - self.grid.loc[i]['min_lat'])/2.
                    central_lon = self.grid.loc[i]['max_long'] - (self.grid.loc[i]['max_long'] - self.grid.loc[i]['min_long'])/2.
                    total_central_lat.append(central_lat)
                    total_central_lon.append(central_lon)
                    
                    #if the number of stations is not 0 for the cell:
                    if self.grid.loc[i]['number_station'] != 0:
                        #defining temporal dataframe to analyze cells
                        cell_month_df = pd.DataFrame({'time': taxis_month})
                        cell_month_df = cell_month_df.set_index(['time'])
                        
                        #for each id_station (j) in the 'indices' attribute of the interest cell:
                        for j in self.grid.loc[i]['indices']:
                            
                            #merging the observations self.df[j] of the station to the local dataframe
                            cell_month_df = pd.merge(cell_month_df, self.df[j], on='time', how='left')
                        
                        if monthly:
                            #saving monthly resolution into a copied dataframe
                            cell_month_copy = cell_month_df.copy()
                            cell_month_copy = cell_month_copy.mean(axis = 1, skipna = True)
                            cell_month_copy = cell_month_copy.to_frame('{},{}'.format(central_lat, central_lon))
                            #merging into monthly dataframe
                            all_filled_cells_monthly = pd.merge(all_filled_cells_monthly, cell_month_copy, on = 'time', how = 'left')
                    
                        #now we cycle to delete years with totally empty measures in the cell. Note that if at least one
                        #station has an observation in the interest month then that measure is considered valid and the
                        #year is not deleted. we append deleted years preventing deliting them again (would raise errors)
                        deleted_years = []
                
                        for index, row in cell_month_df.iterrows():
                            if index.year not in deleted_years:
                                if(all(np.isnan(row.values))):
                                    cell_month_df = cell_month_df.drop([ind for ind in cell_month_df.loc[str(index.year)].index], axis = 0)
                                    deleted_years.append(index.year)
                
                            else:
                                continue

                        #now averaging over rows skipping nan
                        cell_month_df = cell_month_df.mean(axis = 1, skipna =True)
                    
                        #averaging over whole years
                        cell_month_df = cell_month_df.groupby(cell_month_df.index.year).mean()
                        #converting series to dataframe to merge. Note that the name of the columns
                        #will be the central latitude and central longitude of the cell
                        cell_month_df = cell_month_df.to_frame('{},{}'.format(central_lat, central_lon))
                        
                        #merging into the final dataframe to be returned on 'time' columns
                        all_filled_cells_annual_mean = pd.merge(all_filled_cells_annual_mean, cell_month_df, on = 'time', how = 'left')
                                    
                    #if the cell has no stations in it
                    else:
                                        
                        #creating a nan vector of the same length of the taxis
                        nans = [np.nan]*len(taxis)
                    
                        #creating empty cell dataframe and appending it to the final dataframe
                        empty_cell = pd.DataFrame({'time': taxis, '{},{}'.format(central_lat, central_lon): nans})
                        empty_cell = empty_cell.set_index(['time'])
                        all_filled_cells_annual_mean = pd.merge(all_filled_cells_annual_mean, empty_cell, on = 'time', how = 'left')
                        if monthly:
                            nans = [np.nan]*len(taxis_month)
                            empty_cell = pd.DataFrame({'time': taxis_month, '{},{}'.format(central_lat, central_lon): nans})
                            empty_cell = empty_cell.set_index(['time'])
                            all_filled_cells_monthly = pd.merge(all_filled_cells_monthly, empty_cell, on = 'time', how = 'left')
                                                
            if meshed:
                if monthly:
                    return all_filled_cells_annual_mean, all_filled_cells_monthly, np.unique(total_central_lat), np.unique(total_central_lon)
                else:
                    return all_filled_cells_annual_mean, np.unique(total_central_lat), np.unique(total_central_lon)
            else:
                return all_filled_cells_annual_mean

        def to_annual(self, attribute = None, del_incomplete = True):
            """
                Returns a dataset of annual mean temperatures.
                Arguments:
                del_incomplete = True: If true we delete from the dataset years
                with incomplete months to delete bias.
                attribute_df = None: If none then annual aggregation will be done
                on self.df, else we can pass a specified
                attribute of the class.
                
                returns: Dataset with annual mean and annual index
            """
                                
            attribute_df = self.df.copy() if attribute is None else attribute.copy()

            for i in tqdm(attribute_df.columns, desc = 'computing annual df'):
                drop_y = attribute_df[i][np.isnan(attribute_df[i])].index.year.unique()
                for j in attribute_df.index:
                    if j.year in drop_y:
                        attribute_df[i][j] = np.nan
                                                
            attribute_df = attribute_df.groupby(pd.Grouper(freq="Y")).mean()
                                                
            return attribute_df


        def to_anomaly(self, on = 'df', period=['1961', '1990'], annual = True, update = False):
            """
                Creating a dataset of anomalies monthly or yearly from the
                original dataset just subtracting the mean for the interest
                period.
                Arguments:
                on='df' : Attribute as str on which compute anomaly. It has
                to be an istance of the class Full_station.
                period = ['1961', '1990']: standard period of reference on
                which to compute anomalies. default is the standard
                Crutem 30 years refefrence period 1961-1990.
                annual = True: Decide either to return annual resolution
                anomaly or monthly (annual = False). Note that
                user can ask for annual even if 'on' is a monthly
                dataset. Does not work the other way,
                if 'on' is an yearly resoluted dataset.
                update = False: Choose to update the attribute 'on' on which we
                retrieved the data. So if updated=True then the
                object Full_station will be modified permanently
                and Full_station.on will be now an anomaly dataset.
            """
            #retireve from class object the dataset to convert into anomaly if nothing
            #specified anomaly convertion will be done on self.df
            dataset = getattr(self, on)

            #check frequency:
            if dataset.index[1] - dataset.index[0] > pd.Timedelta('2 day') and dataset.index[1] - dataset.index[0] < pd.Timedelta('365 day') :
                frequency = 'M'
            elif dataset.index[1] - dataset.index[0] <= pd.Timedelta('366 day') and dataset.index[1] - dataset.index[0] >= pd.Timedelta('364 day'):
                frequency = 'Y'
            else:
                frequency = 'D'
                print('Cannot compute anomaly daily')
                return
                                                        
            #if the frequency is monthly we need to compute anomaly monthly subtracting at each month
            #the mean of the months in the reference period:
            if frequency == 'M':
                #if annual is not specified:
                if not annual:
                    dataset_interest = dataset[period[0]:period[1]].copy()
                    for i in range(1,13):
                        dataset[dataset.index.month == i] = dataset[dataset.index.month == i] - dataset_interest[dataset_interest.index.month == i].mean()
                else:
                    #computing annual dataset calling to annual function
                    dataset = self.to_annual(attribute = dataset)
                    #subtracting the mean in the interest period
                    dataset = dataset - dataset[period[0]:period[1]].mean()
                                    
            elif frequency == 'Y':
                dataset = dataset - dataset[period[0]:period[1]].mean()
                                            
            if update:
                setattr(self, on, dataset)
            else:
                return dataset





