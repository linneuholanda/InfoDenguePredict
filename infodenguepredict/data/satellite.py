"""
Downloads satelite imagery for temperature and precipitation

This module make requests to the IRI library from Columbia University.

 SOURCES .UCSB .CHIRPS .v2p0 .daily .global
 http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily/.global/.0p05/T/(Jan%202016)/(Nov%202016)/RANGE/X/-44.0/-42.6/RANGE/Y/-23.1/-22.5/RANGE/T/pentadAverage/prcp/T/1460.5/VALUE/%5BX/Y/%5D/palettecolor.tiff?filename=data19810101-05.tiff
"""
import pandas as pd
import os


class LandSurfaceTemperature:
    """
    Images obtained with the following dcode:
          '''
      SOURCES .USGS .LandDAAC .MODIS .1km .8day .version_005 .Terra .SSA .Night .LST
      T (Jan 2016) (Nov 2016) RANGE
      X -44.0 -42.6 RANGE
      Y -23.1 -22.5 RANGE
      T pentadAverage
      '''
    """
    base_url = "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.1km/.8day/.version_005/.Terra/.SSA/.Night/.LST/(Celsius)/unitconvert/T/(Jan%202016)/(Nov%202016)/RANGE/X/{west}/{east}/RANGE/Y/{south}/{north}/RANGE/T/pentadAverage/T/4016.5/4017.5/RANGE/X/{west}/{east}/RANGEEDGES/Y/{south}/{north}/RANGEEDGES/%5BX/Y/%5D/palettecolor.tiff?filename=data2016{month}{sday}-{eday}.tiff"

    def get_5day_average_image(self, west, east, south, north, start_date='20160101', end_date='20161115'):
        """
        Get a five-day average land surface temperature on a square window
        :param west: West bound of the square
        :param east:
        :param south:
        :param north:
        :param start_date:
        :param end_date:
        """
        dates = pd.date_range(start_date, end_date, freq='5D')
        for d in dates:
            stopday = d + 1
            D = str(d).split()[0]
            mes = D.split('-')[1]
            dia = D.split('-')[2]
            url = self.base_url.format(**{'month': mes,
                                        'sday': dia,
                                        'eday': str(stopday).split()[0].split('-')[2],
                                        'west': west,
                                        'east': east,
                                        'north': north,
                                        'south': south
                                        })
            os.system("wget '{}'".format(url))


class Precipitation:
    """
    Data is obtained with the following IRI code:

      '''
      SOURCES .UCSB .CHIRPS .v2p0 .daily .global .0p05
      T (Jan 2016) (Nov 2016) RANGE
      X -44.0 -42.6 RANGE
      Y -23.1 -22.5 RANGE
      T pentadAverage
      prcp prcp
      '''
    """
    base_url = "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily/.global/.0p05/T/(Jan%202016)/(Nov%202016)/RANGE/X/{west}/{east}/RANGE/Y/{south}/{north}/RANGE/T/pentadAverage/prcp/prcp/T/1460.5/VALUE/%5BX/Y/%5D/palettecolor.tiff?filename=data19810101-05.tiff"

    def get_5day_average_image(self, west, east, south, north, start_date='20160101', end_date='20161115'):
        """
        Get a five-day average land surface temperature on a square window
        :param west: West bound of the square
        :param east:
        :param south:
        :param north:
        :param start_date:
        :param end_date:
        """
        dates = pd.date_range(start_date, end_date, freq='5D')
        for d in dates:
            stopday = d + 1
            D = str(d).split()[0]
            mes = D.split('-')[1]
            dia = D.split('-')[2]
            url = self.base_url.format(**{'month': mes,
                                        'sday': dia,
                                        'eday': str(stopday).split()[0].split('-')[2],
                                        'west': west,
                                        'east': east,
                                        'north': north,
                                        'south': south
                                        })
            os.system("wget '{}'".format(url))
