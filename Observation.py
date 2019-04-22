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


