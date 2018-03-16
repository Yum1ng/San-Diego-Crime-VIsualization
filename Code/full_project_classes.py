import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from datetime import datetime

import string
from gmplot import gmplot
import googlemaps

# Put a valid google service key here to use googlemap
gmaps = googlemaps.Client(key='___key_removed___')

pd.set_option('max_columns', None)
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
rcParams['figure.figsize'] = 12, 10


class CrimeAnalysis:
    def __init__(self):
        """
        Initialize variables
        """
        self.fn = 'crime.txt'
        self.df = self.build_df()
        self.mc_fn = 'monthly_crime.xls'
        self.mc = self.build_mc()
        self.cols = [u'murder', u'rape', u'armed_robbery', u'strong_arm_robbery', u'aggravated_assault',
                     u'total_violent_crime',
                     u'residential_burglary', u'non_residential_burglary', u'total_burglary',
                     u'theft_above_400', u'theft_below_400', u'total_thefts',
                     u'motor_vehicle_theft', u'total_property_crime', u'total']
        self.years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def build_df(self):
        """
        Create initial data.sandiego.gov dataframe
        :return:
        """
        df = pd.read_csv(self.fn)
        df.columns = ['agency', 'crime', 'time', 'addr', 'zip', 'community']
        df['crime'] = df['crime'].astype(str)
        return df

    def build_mc(self):
        """
        Create initial ARJIS dataframe
        :return:
        """
        mc = pd.read_excel(self.mc_fn).transpose()
        return mc

    @staticmethod
    def time_fix(row):
        """
        Helper method for use with apply()
        build datetime obj from time string
        :param row:
        :return:
        """
        t = str(row['time'])
        return datetime.strptime(t, "%m/%d/%Y %H:%M:%S")

    @staticmethod
    def add_doy(row):
        """
        Helper method for use with apply()
        get day of year of time obj
        :param row:
        :return:
        """
        d = row['time_fix']
        return d.timetuple().tm_yday

    @staticmethod
    def add_month(row):
        """
        Helper method for use with apply()
        get month from time obj
        :param row:
        :return:
        """
        m = row['time_fix']
        return m.month

    @staticmethod
    def add_hour(row):
        """
        Helper method for use with apply()
        Get the hour from the time obj
        :param row:
        :return:
        """
        h = row['time_fix']
        return h.hour

    @staticmethod
    def type_find(row):
        """
        Helper method for use with apply()
        Parse the crime description column of df to find the type
        :param row:
        :return:
        """
        t = str(row['crime']).lower()
        if 'firearm' in t or 'ammunition' in t or 'shoot' in t:
            return 'gun'
        if 'controlled' in t or 'contr' in t or 'drug' in t or 'paraphernalia' in t or 'cntl' in t:
            return 'drug'
        if 'theft' in t or 'burglary' in t or 'robbery' in t or 'obtain money' in t:
            return 'theft'
        if 'drunk' in t or 'liquor' in t or 'open container' in t or 'alcohol' in t or 'alc' in t:
            return 'alcohol'
        if 'marijuana' in t or 'cannabis' in t or 'weed' in t:
            return 'weed'
        if 'weapon' in t or 'metal knuckles' in t or 'leaded cane' in t or 'shuriken' in t or 'knife' in t or 'dagger' in t:
            return 'weapons'
        if 'sex' in t or 'rape' in t or 'intimate' in t or 'indecent exposure' in t or 'obscene' in t or 'prostitution' in t:
            return 'sexual'
        if 'assault' in t or 'battery' in t:
            return 'assault'
        if 'resist' in t:
            return 'resisting'
        if 'shoplifting' in t:
            return 'shoplifting'
        if 'fraud' in t or 'defraud' in t or 'personate' in t:
            return 'fraud'
        if 'vandalism' in t:
            return 'vandalism'
        if 'elder' in t:
            return 'elder abuse'
        if 'get credit' in t or 'personal identific' in t:
            return 'identity theft'
        if 'terrorize' in t or 'terrorist' in t:
            return 'terrorism'
        if 'animal' in t:
            return 'animal'
        if 'child' in t or 'minor' in t:
            return 'child'
        if 'tamper' in t or 'carjacking' in t:
            return 'vehicle tampering'
        if 'arson' in t:
            return 'arson'
        else:
            return 'none'

    def fix_df(self):
        """
        Clean up the data.sandiego.gov dataset
        :return:
        """
        df = self.df
        df['time_fix'] = df.apply(self.time_fix, axis=1)
        df['doy'] = df.apply(self.add_doy, axis=1)
        df['month'] = df.apply(self.add_month, axis=1)
        df['hour'] = df.apply(self.add_hour, axis=1)
        df['type'] = df.apply(self.type_find, axis=1)
        return df

    def fix_mc_df(self):
        """
        Clean up the ARJIS dataset
        :return:
        """
        mc = self.mc
        mc.columns = mc.iloc[1]
        mc.drop(mc.index[0], inplace=True)  # drop crime row
        mc.drop(mc.index[0], inplace=True)  # drop sort_order row
        mc.drop('Jan / 2018', inplace=True)
        mc.drop('Feb / 2018', inplace=True)
        mc.drop('Total', inplace=True)
        # mc
        new = pd.MultiIndex.from_product([self.years, self.months], names=['years', 'months'])
        # new.values = mc.values
        # new.columns = cols
        # new
        mc.set_index(new, inplace=True)
        mc.columns = self.cols
        # mc.xs('Jan', level=1, drop_level=False)['total']
        return mc

    def fix_data(self):
        """
        Clean all data
        :return:
        """
        self.df = self.fix_df()
        self.mc = self.fix_mc_df()

    def build_sd_trends(self):
        """
        Plot the time of day and crime types for San Diego county as a whole
        Create the figure, add the axes, and set the axes parameters
        :return:
        """
        df = self.df
        fig, axarr = plt.subplots(1, 2, figsize=(18, 8))
        plt.suptitle("Crime Trends in San Diego County")

        # sns.countplot(y='type', data=df[df['hour'] == 1], order=df['type'].value_counts().index, color='green')
        sns.set(font_scale=1.5)
        ax = sns.countplot(y='type', data=df, order=df['type'].value_counts().index, color='green', ax=axarr[0])
        ax.title.set_text("Crime Sorted by Type")
        ax.set_xlabel('Count')
        ax.set_ylabel('Type')

        # sns.countplot(x='hour', data=df.drop(df[df['type'] == 'identity theft'].index))
        # sns.countplot(x='hour', data=df[df['type'] == 'theft'])
        ax2 = sns.countplot(x='hour', data=df, ax=axarr[1])
        ax2.title.set_text("Crime by Hour of Day")
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Count')

    def build_dt_trends(self):
        """
        Plot the time of day and type for crimes in Downtown San Diego
        Create the figure, add the axes, and set the axes parameters
        :return:
        """
        df = self.df
        dt = df[df['zip'] == 92101.0]

        fig, axarr = plt.subplots(1, 2, figsize=(18, 8))
        plt.suptitle("Crime Trends Downtown")

        sns.set(font_scale=1.5)
        ax = sns.countplot(y='type', data=dt, order=df['type'].value_counts().index, color='green', ax=axarr[0])
        ax.title.set_text("Crime Sorted by Type")
        ax.set_xlabel('Count')
        ax.set_ylabel('Type')

        ax2 = sns.countplot(x='hour', data=dt, ax=axarr[1])
        ax2.title.set_text("Crime by Hour of Day")
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Count')

    def build_pb_trends(self):
        """
        Plot the time of day and type for crimes in Pacific Beach/Mission Beach
        Create the figure, add the axes, and set the axes parameters
        :return:
        """
        df = self.df
        pb = df[df['zip'] == 92109.0]

        fig, axarr = plt.subplots(1, 2, figsize=(18, 8))
        plt.suptitle("Crime Trends in Pacific Beach and Mission Beach")

        sns.set(font_scale=1.5)
        ax = sns.countplot(y='type', data=pb, order=df['type'].value_counts().index, color='green', ax=axarr[0])
        ax.title.set_text("Crime Sorted by Type")
        ax.set_xlabel('Count')
        ax.set_ylabel('Type')

        ax2 = sns.countplot(x='hour', data=pb, ax=axarr[1])
        ax2.title.set_text("Crime by Hour of Day")
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Count')

    def build_yearly_regression(self):
        """
        Plot regression lines for each year over the last 10 years showing crime throughout the months
        Create the figure, add the axes, and set the axes parameters
        :return:
        """
        mc = self.mc
        mc['m'] = np.tile(np.arange(1, 13), 10)
        # mc.reset_index()

        lm = sns.lmplot(data=mc.reset_index(), x='m', y='total', hue='years',
                        order=7, truncate=True, size=10, ci=None)
        # order=7, truncate=True, size=10, ci=None, scatter_kws={"s": 0})
        ax = lm.axes[0, 0]
        ax.set_ylim(4000, 9000)
        ax.title.set_text("Crime by Year and Month")
        ax.set_xlabel('Month')
        ax.set_ylabel('Crimes Committed')

    def build_time_trends(self):
        """
        Show bar graphs of yearly trends
        Create the figure, add the axes, and set the axes parameters
        :return:
        """
        mc = self.mc
        months = self.months
        fig, axarr = plt.subplots(1, 2, figsize=(18, 8))
        plt.suptitle("Crime Trends Over Time")

        ax = sns.barplot(y=mc['total'].unstack(level=0).sum().reset_index().drop('years', axis=1).squeeze().values,
                         x=np.arange(2008, 2018), ax=axarr[0])

        ax.title.set_text("Crime by Year")
        ax.set_xlabel('Year')
        ax.set_ylabel('Crimes Committed')

        ax2 = sns.barplot(y=mc['total'].unstack(level=0).reindex(months).sum(axis=1).values, x=months,
                          ax=axarr[1])
        ax2.title.set_text("Crime by Month")
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Crimes Committed')

    def build_trends(self):
        """
        Show trends over time for different categories and an overall regression line
        Create the figure, add the axes, and set the axes parameters
        :return:
        """
        mc = self.mc
        sns.set(font_scale=1.2)
        fig, axarr = plt.subplots(2, 2, figsize=(16, 10))
        plt.suptitle("Crime Patterns and Categories")

        total = sns.barplot(y=mc['total'].unstack(level=0).sum().reset_index().drop('years', axis=1).squeeze().values,
                            x=np.arange(2008, 2018), ax=axarr[0][0])
        total_vc = sns.barplot(
            y=mc['total_violent_crime'].unstack(level=0).sum().reset_index().drop('years', axis=1).squeeze().values,
            x=np.arange(2008, 2018), ax=axarr[1][0])
        total_pc = sns.barplot(
            y=mc['total_property_crime'].unstack(level=0).sum().reset_index().drop('years', axis=1).squeeze().values,
            x=np.arange(2008, 2018), ax=axarr[1][1])
        reg = sns.regplot(x=np.arange(1, 121), y=mc['total'].values, order=10, n_boot=1000, truncate=True,
                          ax=axarr[0][1])

        # axarr[0][0] = total
        # axarr[0][1] = total_vc
        # axarr[1][0] = total_t
        # axarr[1][1] = total_pc
        total.title.set_text('Overall Crime Count')
        reg.title.set_text('Seasonal Trends\nHigh Order Polynomial Regression')
        total_vc.title.set_text('Violent Crime Count')
        total_pc.title.set_text('Property Crime Count')


class Heatmap:
    def __init__(self, date):
        print("Reading data...")
        # read the data
        df = pd.read_csv('crime.txt')
        df.columns = ['agency', 'crime', 'time', 'addr', 'zip', 'community']
        df['crime'] = df['zip'].astype(str)
        self.date = date
        self.crime_lat_lon_list = self.get_last_week_crime_data(date, df)

    def check_CA_lat_lon(self, lat, lon):
        """
        # check whether the lat, lon is near San Diego, can shrink this to ((32, 34), (-118, -115))
        :param lon: lon of the address
        :param lat: lat of the address
        :return: True if is near SD, False othewise
        """
        if (32 <= lat <= 35) and (-118 <= lon <= -115):
            return True
        else:
            return False

    def get_last_week_crime_data(self, date, df):
        """
        Get crime data just from last week to limit api calls
        :param date: string format for date
        :param df: the dataframe
        :return: a list of (lag, lon) tuples that can be plot using google-map-heatmap
        """
        from datetime import datetime
        addr = []
        addr_dict = {}

        # clean the data first, filter out address that cannot be found or invalid
        for i in range(len(df['time'])):
            if len(df['time'][i]) == 0 or len(df['addr'][i]) == 0:
                continue
            dif = (datetime.strptime(df['time'][i], "%m/%d/%Y %H:%M:%S") - datetime.strptime(date,
                                                                                             "%m/%d/%Y %H:%M:%S")).days
            if -7 <= dif < 0:
                if df['addr'][i] in addr_dict:
                    addr_dict[df['addr'][i]] += 1
                else:
                    addr_dict[df['addr'][i]] = 1
                addr.append(df['addr'][i])
        res_list = []
        # for every address, get the lat, lon of it by using google api, this takes 10 minutes for 2000 points
        for key in addr_dict:
            geocode_result = gmaps.geocode(key)
            if 0 == len(geocode_result):
                continue
            curr_lat = geocode_result[0]["geometry"]["location"]['lat']
            curr_lon = geocode_result[0]["geometry"]["location"]['lng']
            if self.check_CA_lat_lon(curr_lat, curr_lon):
                print(curr_lat, curr_lon, key)
                latlon = (curr_lat, curr_lon)
                for i in range(addr_dict[key]):
                    res_list.append(latlon)
        print("all valid near SD crime number in the past week", len(res_list))
        return res_list

    def find_max_inten(self, date, df):
        """
        Get the max_crime number of the crimes in a address, this is a function for internal use, will not be called
        """
        addr = []
        addr_dict = {}
        for i in range(len(df['time'])):
            if 0 == len(df['time'][i]) or len(df['addr'][i]) == 0:
                continue
            dif = (datetime.strptime(df['time'][i], "%m/%d/%Y %H:%M:%S") - datetime.strptime(date,
                                                                                             "%m/%d/%Y %H:%M:%S")).days
            if -7 <= dif < 0:
                if df['addr'][i] in addr_dict:
                    addr_dict[df['addr'][i]] += 1
                else:
                    addr_dict[df['addr'][i]] = 1
                addr.append(df['addr'][i])
        print("all addresses: ", len(addr))
        print("all unique address: ", len(addr_dict))
        # print(addr_dict)
        res_list = []
        max_inten = 0
        max_dir = ""
        for key in addr_dict:
            if key == '0  BLOCK GBDF':
                continue
            if addr_dict[key] > max_inten:
                max_inten = addr_dict[key]
                max_dir = key
        print("max_inten is : ", max_inten)
        print("max_inten address is : ", max_dir)

    def generate_past_week_html(self, crime_lat_lon_list, center_lat=32.7269669, center_lon=-117.1647094):
        """
        Generate a html from the data gathered in the past week, the center of the map default to SD Downtown,
        but can be adjust to user's preference
        """
        from gmplot import gmplot
        gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, 13)

        # Center of the map
        gmap.marker(center_lat, center_lon, 'cornflowerblue')
        # Scatter points
        top_attraction_lats, top_attraction_lons = zip(*crime_lat_lon_list)
        gmap.heatmap(top_attraction_lats, top_attraction_lons, threshold=100, radius=20, gradient=None, opacity=0.5,
                     dissipating=True)
        # Draw
        gmap.draw("last_week_crime_heatmap.html")

    def get_zipcode_crime_map(self, zipcode):
        """
        Generate a html heatmap for the zipcode.
        """
        from uszipcode import ZipcodeSearchEngine
        search = ZipcodeSearchEngine()
        zip_lat = search.by_zipcode(zipcode)['Latitude']
        zip_lon = search.by_zipcode(zipcode)['Longitude']
        self.generate_past_week_html(self.crime_lat_lon_list, center_lat=zip_lat, center_lon=zip_lon)

    def get_address_crime_map(self, address):
        """
        Generate a html heatmap for a specific addresses
        address: a string, of the spefic address, for example, "3869 Miramar Street"
        """
        geocode_result = gmaps.geocode(address)
        if len(geocode_result) == 0:
            print("cannot get the address")
            return
        curr_lat = geocode_result[0]["geometry"]["location"]['lat']
        curr_lon = geocode_result[0]["geometry"]["location"]['lng']
        if not self.check_CA_lat_lon(curr_lat, curr_lon):
            print("cannot get the address")
            return
        self.generate_past_week_html(self.crime_lat_lon_list, center_lat=curr_lat, center_lon=curr_lon)


class Density:
    def __init__(self):
        self.df = pd.read_csv('crime.txt')
        self.df.columns = ['agency', 'crime', 'time', 'addr', 'zip', 'community']

    # check if the lat, lon is valid
    def check_SD_lat_lon(self, lat, lon):
        if not (not (32 <= lat <= 33.4) or not (-118 <= lon <= -115)):
            return True
        else:
            return False

    # check if the zipcode is valid
    def isvalid(self, zipcode):
        from uszipcode import ZipcodeSearchEngine
        search = ZipcodeSearchEngine()
        if zipcode < 90000 or zipcode > 99999 or zipcode == 91980:
            return False
        lat = search.by_zipcode(zipcode)['Latitude']
        lon = search.by_zipcode(zipcode)['Longitude']
        return self.check_SD_lat_lon(lat, lon)

    def get_xy_crimeRate_popDensity(self, df):
        """
        Get the relationship between population density and Crime rate, plot in x, y coordinates form
        """
        from uszipcode import ZipcodeSearchEngine
        search = ZipcodeSearchEngine()

        crime_rate = []
        annual_wage = []
        population_dens = []
        valid_zip = df['zip'].unique().astype(np.int)
        crime_num = df.groupby(['zip']).size()
        for i in valid_zip:
            if not self.isvalid(i):
                continue
            else:
                wage = search.by_zipcode(i)['Wealthy']
                population_den = search.by_zipcode(i)['Density']
                rate = crime_num.get(i) / 180.0
                if wage is None:
                    continue
                if rate > 20:
                    print("highest rate: ", i, "rate: ", rate, ", wage: ", wage, " pop desity: ", population_den)
                if population_den > 12000:
                    print("highest density: ", i, "rate: ", rate, ", wage: ", wage, " pop desity: ", population_den)
                # a way of measuring good community - dense population, but small crime rate
                if population_den > 10000 and rate < 4:
                    print("low wage and low crime: ", i, "rate: ", rate, ", wage: ", wage, " pop desity: ",
                          population_den)
                annual_wage.append(int(wage) * 0.01)
                crime_rate.append(rate)
                population_dens.append(population_den)
                # print(curr_population)
        plt.scatter(crime_rate, population_dens, s=annual_wage, alpha=0.4)
        plt.xlabel('Crime Numbers (times / day)')
        plt.ylabel('Population Density (people / km$^2$)')
        for income in [10000, 20000, 40000]:
            plt.scatter([], [], c='k', alpha=0.4, s=income * 0.01, label=str(income) + ' $ / year')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Annual wage')
        plt.title('Crime Rate and Population Density for San Diego Area Zipcodes')
        plt.show()


if __name__ == '__main__':
    ca = CrimeAnalysis()
    ca.fix_data()

    hm = Heatmap('2/5/2018 16:30:00')
    hm.get_address_crime_map('9500 Gilman Dr, La Jolla, CA 92093')

    d = Density()
    d.get_xy_crimeRate_popDensity(d.df)
