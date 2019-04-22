import numpy as np
import pandas as pd
import datetime as dtm
import re
import math as mt
import random
import io
from tqdm import tqdm
import xarray as xr

class Model():
    """
        Reading and analizing historical models.
    """
    
    def __init__(self, model_path):
        """
            Initialize with model name and model path.
            """
        self.model_name = model_path.split('/')[-1]
        self.model_path = model_path
    
    def reader(self):
        
        """
            Reading the model saving the xarray dataset of
            celsius temperatures to model and to recovery attributes
        """
        #opening model with xarray
        model = xr.open_dataset(self.model_path)
        
        #if the dataset is a temperature dataset, converting temp to celsius
        if "ts" in self.model_name[:4]:
            model = model.ts -273.15
        
        self.model = model
        self.recover = model
    
    def recovery(self):
        """
            Recover attributes when dataset was created
        """
        #recovering data from when they were red:
        self.model = self.recover
    
    def selecting(self, latitude, longitude, out_path = None, plot = False, annual = False, Save = False):
        """
            Selecting from the model a defined latitude and
            longitude (ex: a city).
            Arguments:
            latitude: latitude in [-90,90] of the place
            longitude: longitude in [-180, 180] of the place
            out_path = None: out_path to save plot
            plot = False: plotting returns position on a map
            and time series
            annual = False: Computing also annual time series
            of the place in interest
            Save = False: Decide to save the current canvas
            
            return: place (monthly resolution array)
            if annual:
            return place, place_annual (annual resolution array)
            """
        #Selecting a place given latitude and longitude
        place = self.model.sel(lon=longitude, lat=latitude, method = 'nearest')
        
        if annual:
            place_annual_mean = place.groupby('time.year').mean('time')
        
        if plot:
            
            #plotting coordinates on a map
            fig = plt.figure(figsize=(20,5))
            fig.suptitle("Average monthly time series for {}/{}".format(latitude, longitude))
            
            if annual:
                ax = fig.add_subplot(1,3,1,projection=ccrs.PlateCarree())
            else:
                ax = fig.add_subplot(1,2,1,projection=ccrs.PlateCarree())
            ax.set_global()
            ax.stock_img()
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax.add_feature(cfeature.LAND)
            ax.plot(longitude, latitude, color='red', linewidth=10, marker='o', markersize=3, transform=ccrs.Geodetic())
            
            #plotting monthly time series
            if annual:
                ax1 = fig.add_subplot(1,3,2)
            else:
                ax1 = fig.add_subplot(1,2,2)
            
            place.to_series().plot(c = 'orange')
            ax1.set_xticklabels([t for i,t in enumerate(ax1.get_xticklabels())], rotation = 45, fontsize = 5)
            
            #if annual plots the annual average of the place in interest
            if annual:
                ax2 = fig.add_subplot(1,3,3)
                place_annual_mean.to_series().plot(c = 'red')
            
            #ax1.set_xlabel('time')
            ax1.set_ylabel('Mean temperature [°C]')

        if Save:
            try:
                fig.savefig(out_path + '/model_{}:{}.pdf'.format(latitude, longitude))
            
            except:
                print('Failed in saving plot. Check if Save = True and if out_path is passed')

                    if annual:
            return place_annual_mean, place
        else:
            return place



    def slicing(self, sliced = False, lat = False, lon = False, time = False, update = False):
        """
            Slices a region of space from the model.
            Arguments:
            sliced = False: we can pass the method a sliced dataframe
            lat = False: lat interval as [ , ]
            lon = False: lon interval as [ , ]
            time = False: time interval as ["", ""]
            update = False: if update self.model is
            updated with the slice defined
            
            if not update:
            return sliced
            """
        if lat:
            if update:
                self.model = self.model.sel(lat=slice(lat[0],lat[1]))
            else:
                if sliced:
                    sliced = sliced.sel(lat=slice(lat[0],lat[1]))
                else:
                    sliced = self.model.sel(lat=slice(lat[0],lat[1]))
        
        if lon:
            if update:
                self.model = self.model.sel(lon=slice(lon[0], lon[1]))
            else:
                if sliced:
                    sliced = sliced.sel(lon=slice(lon[0], lon[1]))
                else:
                    sliced = self.model.sel(lon=slice(lon[0], lon[1]))

        if time:
            if update:
                self.model = self.model.sel(time=slice(time[0],time[1]))
            else:
                if sliced:
                    sliced = sliced.sel(time=slice(time[0],time[1]))
                else:
                    sliced = self.model.sel(time=slice(time[0],time[1]))
    
        if not update:
            return sliced


    def averaging(self, Global = False, update_df = False):
    """
        #Mean on the region, computing the mean for each time step. antartica_region array
        #will just contain the variable ts (averaged over lat and lon) and time
        
        #if update condition is true then the model attribute is updated and no other istance
        #is created into the class. Otherwise another attribute 'annual_mean' is created.
    """
        if update_df:
            self.model = self.model.groupby('time.year').mean('time')
            #renaming the time index from 'year' to 'time'
            self.model['time'] = self.model['year']
                self.model = self.model.drop('year')
        else:
            self.annual_mean = self.model.groupby('time.year').mean('time')
            #renaming the time index from 'year' to 'time'
            self.annual_mean['time'] = self.annual_mean['year']
            self.annual_mean = self.annual_mean.drop('year')
    
        #if global mean is computed across all latitudes and longitudes
        if Global:
            self.global_mean = self.model.mean(dim = ['lat','lon'])
            self.global_mean = self.global_mean.groupby('time.year').mean('time')

    def to_dataset(self, skip = 1, meshed = False, , anomaly = False):
        """
            Creates a pandas dataframe out of the model
            after loading it. The dataframe is saved into self.df
            Arguments:
            skip = 1: Skipping latitudes from the xarray
            latitude/longitude when loaded. default = 1,
            nothing is skipped.
            meshed = False: if True it returns also the coordinates
            of the central latitudes coherently with
            skipping option.
            anomaly = False: if an anomaly attribute is present in
                            the object then this will create an
                            anomaly dataframe
            
            return: saves dataframe into self.df.
            Returns nothing if meshed returns the unique
            values of central_latitudes and central_longitudes
            
            
        """
        if len(self.model.lat)%skip != 0 and len(self.model.lon)%skip != 0:
            print('skip option will miss some values')

        #creating a pandas dataset out of the model
        if not anomaly:
            taxis = self.model.time
            df = pd.DataFrame({'time': taxis})
            df = df.set_index(['time'])
            
            central_lat = []
            central_lon = []
            
            for i in tqdm(range(0, len(self.model.lat), skip)):
                for j in range(0, len(self.model.lon), skip):
                    
                    #creating local dataframe of the cell with temperatures
                    cell_df = pd.DataFrame({'time': taxis, (float(self.model.lat[i]),float(self.model.lon[j])-180): self.model[:,i,j]})
                    
                    #merging
                    df = pd.merge(df, cell_df, on = 'time', how = 'left')
                    
                    #saving longitudes and latitudes
                    central_lat.append(float(self.model.lat[i]))
                    central_lon.append(float(self.model.lon[j])-180)
                
                df = df.set_index(['time'])

            self.df = df

            if meshed:
                return np.unique(central_lat), np.unique(central_lon)

        else:
    
            taxis = self.anomaly.year
                df = pd.DataFrame({'time': taxis})
                df = df.set_index(['time'])
                
                central_lat = []
                central_lon = []
                
                for i in tqdm(range(0, len(self.anomaly.lat), skip)):
                    for j in range(0, len(self.anomaly.lon), skip):
                        
                        #creating local dataframe of the cell with temperatures
                        cell_df = pd.DataFrame({'time': taxis, (float(self.anomaly.lat[i]),float(self.anomaly.lon[j])-180): self.anomaly[:,i,j]})
                        
                        #merging
                        df = pd.merge(df, cell_df, on = 'time', how = 'left')
                        
                        #saving longitudes and latitudes
                        central_lat.append(float(self.anomaly.lat[i]))
                        central_lon.append(float(self.anomaly.lon[j])-180)
                    
                    df = df.set_index(['time'])
            
                self.anomaly_df = df
                
                if meshed:
                    return np.unique(central_lat), np.unique(central_lon)

    def model_anomaly(self, ref = ['1961', '1990'], default = 'model'):
        """
           computes and return an anomaly xarray with respect to a user given time period.
           Arguments:
           ref = ['1961', '1990']: reference time period to compute anomalies on
           default = 'model': attribute on which to compute the anomaly
        """
    
        if default == 'model':
            anomaly = self.model
            sliced = self.slicing(time = ['1961', '1990'])
            sliced_annual_mean = sliced.groupby('time.year').mean('time')
            sliced_annual_mean = sliced_annual_mean.mean('year')
            
            annual_mean_df = anomaly.groupby('time.year').mean('time')
            anomaly_df = annual_mean_df - sliced_annual_mean
            
            return anomaly_df


