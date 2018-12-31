# Create Python file which will be used as an input for the Dataflow job

from __future__ import absolute_import

import pandas as pd
import matplotlib
matplotlib.use('Agg')
from google.cloud import bigquery
import matplotlib.pyplot as plt
import io
from io import StringIO
import base64
import gc
import argparse
import logging
import numpy as np
from itertools import compress as compress
import apache_beam as beam
from apache_beam.pvalue import AsIter, AsSingleton
from apache_beam.pipeline import PipelineOptions
from apache_beam.pipeline import SetupOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
  
class GenerateImage(beam.DoFn):
    '''
      This is the primary function that generates both a PlaceType Image and a PlaceSubType Image.
      
      PlaceType Image:
        X-axis: represents OSM place types (i.e. restaurant, office building)
        Y-axis: represents individual time block of "stop times" at a place in 5 minute increments (up to 2 hrs)
        Intensity: the intensity of the grid square is the number of times that the vehicle stopped at the place for the defined amount of time (over the course of 1 year)
        
      PlaceSubType Image:
        X-axis represents OSM place sub types (i.e. fast food restaurant, municipal office)
        Y-axis: same as above
        Intensity: same as above
        
      Inputs:
        Row: An individual row of data consists of a HardwareId (which is the grouped key), and ALL trips for the year at an OSM-correlated location
        dfFilteredPlaces: a list of all valid locations from which we will pull (we don't want all OSM locations like "street lamp" because those aren't predictive of industry)
      
    '''
    def process(self, row, dfFilteredPlaces):

        # Pulls in all data from the row and puts it into a dataframe of multiple trips (stop times)
        dfVehicleData = pd.DataFrame(data=[x for x in row[1]], columns=['HardwareId',
                                                               'StopMinutes',
                                                               'PlaceType',
                                                               'PlaceSubType',
                                                               'VehicleType',
                                                               'WeightClass',
                                                               'Industry',
                                                               'Carrier',
                                                               'GovVocation'])

        dfVehicleData['HardwareId'] = dfVehicleData['HardwareId'].astype(str)
        dfVehicleData['StopMinutes'] = dfVehicleData['StopMinutes'].astype(float)
        dfVehicleData['VehicleType'] = dfVehicleData['VehicleType'].astype(str)
        dfVehicleData['WeightClass'] = dfVehicleData['WeightClass'].astype(str)
        dfVehicleData['Industry'] = dfVehicleData['Industry'].astype(str)
        dfVehicleData['Carrier'] = dfVehicleData['Carrier'].astype(str)
        dfVehicleData['GovVocation'] = dfVehicleData['GovVocation'].astype(str)
        dfVehicleData["PlaceTypeCombined"] = (dfVehicleData["PlaceType"] + "-" + dfVehicleData["PlaceSubType"]).astype('category')
                     
        df = dfVehicleData.iloc[[0]]
        for i, row in df.iterrows():
          # Gets the number of elements/data points used to generate each image type
          numPlaceTypeElements = GetNumElements(dfVehicleData, dfFilteredPlaces, 'PlaceType')
          numPlaceSubTypeElements = GetNumElements(dfVehicleData, dfFilteredPlaces, 'PlaceTypeCombined')
          
          # Place Type and Sub Type Image Generation (base64-encoded images)
          placeTypeImage = GeneratePlaceTypeImage(dfVehicleData, dfFilteredPlaces)
          placeSubTypeImage = GeneratePlaceSubTypeImage(dfVehicleData, dfFilteredPlaces)
          newRow = {
                 "VehicleType": row["VehicleType"],
                 "WeightClass": row["WeightClass"],
                 "Industry": row["Industry"],
                 "Carrier": row["Carrier"],
                 "GovVocation": row["GovVocation"],
                 "PlaceTypeImage": str(base64.standard_b64encode(placeTypeImage)).replace("b'","").replace("'",""),
                 "PlaceSubTypeImage": str(base64.standard_b64encode(placeSubTypeImage)).replace("b'","").replace("'",""),
                 "NumPlaceTypeElements": numPlaceTypeElements,
                 "NumPlaceSubTypeElements": numPlaceSubTypeElements}
          gc.collect()
          yield newRow

def GetNumElements(df, dfFilteredPlaces, placeType):
    numElements = 0
    xAxis = dfFilteredPlaces[placeType].cat.categories.tolist()
    for index, row in df.iterrows():
      if row[placeType] in xAxis:
        numElements += 1
    return numElements
                                                    
def GeneratePlaceTypeImage(df, dfFilteredPlaces):
    yAxis = np.arange(0,125,5)
    xAxis = dfFilteredPlaces["PlaceType"].cat.categories.tolist()
    heatmap = np.empty((len(yAxis), len(xAxis)))
    heatmap[:] = 0
    for index, row in df.iterrows():
      if row['StopMinutes'] > 120:
        interval = 24
      else:
        interval = int(row['StopMinutes'] / 5)
      # We only made a listing of 479 places where over the course of the year there were >= 1000 stops
      if row['PlaceType'] in xAxis:
        heatmap[interval, xAxis.index(row['PlaceType'])] += 1

    # temporarily save image to buffer
    buf = io.BytesIO()
    plt.switch_backend('Agg') #Important --> this ensure writing to memory and not display (which isn't supported in Beam)
    plt.set_cmap('gray_r')
    fig = plt.figure()
    im = plt.imshow(heatmap, interpolation='nearest')
    plt.axis('off')
    plt.autoscale(tight=True)
    plt.savefig(buf, format='png', bbox_inches='tight')
    buffer = buf.getvalue()
    buf.close()
    plt.close(fig)
    return buffer

def GeneratePlaceSubTypeImage(df, dfFilteredPlaces):
    yAxis = np.arange(0,125,5)
    xAxis = dfFilteredPlaces["PlaceTypeCombined"].cat.categories.tolist()
    heatmap = np.empty((len(yAxis), len(xAxis)))
    heatmap[:] = 0
    for index, row in df.iterrows():
      if row['StopMinutes'] > 120:
        interval = 24
      else:
        interval = int(row['StopMinutes'] / 5)
      # We only made a listing of 479 places where over the course of the year there were >= 1000 stops
      if row['PlaceTypeCombined'] in xAxis:
        heatmap[interval, xAxis.index(row['PlaceTypeCombined'])] += 1

    # temporarily save image to buffer
    buf = io.BytesIO()
    plt.switch_backend('Agg') #Important --> this ensure writing to memory and not display (which isn't supported in Beam)
    plt.set_cmap('gray_r')
    fig = plt.figure()
    im = plt.imshow(heatmap, interpolation='nearest')
    plt.axis('off')
    plt.autoscale(tight=True)
    plt.savefig(buf, format='png', bbox_inches='tight')
    buffer = buf.getvalue()
    buf.close()
    plt.close(fig)
    return buffer

def GetFilteredPlaces():
    # Filters the place types that we will use (i.e. fast food restaurant, etc.)  We only want to use place types where there have been
    # a number of visits (i.e. greater than 1000 within a one year period)

    filteredPlaces = '''

    select Place1TypeKey as PlaceType, Place1TypeValue as PlaceSubType from `geotab-bi.Project_IndustryClassification.Trips_FivePlaces` where Place1TypeKey is not null
    group by Place1TypeKey, Place1TypeValue
    having count(1) >= 1000
    ORDER BY Place1TypeKey, Place1TypeValue

    '''

    # Defines a listing of acceptable Place Types based on the aforementioned query

    dfFilteredPlaces = pd.read_gbq(filteredPlaces, project_id='geotab-bi', dialect='standard')
    dfFilteredPlaces["PlaceTypeCombined"] = (dfFilteredPlaces["PlaceType"] + "-" + dfFilteredPlaces["PlaceSubType"]).astype('category')
    dfFilteredPlaces["PlaceType"] = dfFilteredPlaces["PlaceType"].astype('category')
    return dfFilteredPlaces
          
def run(argv = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_images', required = True, help = 'Destination for Images')
    known_args, pipeline_args = parser.parse_known_args(argv)
    dfFilteredPlaces = GetFilteredPlaces()
    
    # Pulls in all data ~400M+ rows from 177K vehicles aligned with OSM data and the top five closest locations to a trip stop
    allData = '''
    SELECT
      HardwareId,
      round(stopduration/60,2) as StopMinutes,
      Place1TypeKey as PlaceType,
      Place1TypeValue as PlaceSubType,
      VehicleType,
      WeightClass,
      IHS_CompanyVocation as Industry,
      IHS_CarrierType as Carrier,
      IHS_GovVocation as GovVocation
    FROM
      `geotab-bi.Project_IndustryClassification.Trips_FivePlaces`
    '''
    
    # Defines the Apache Beam pipeline
    with beam.Pipeline(argv = pipeline_args) as p:
        # Step 1: Pulls in all vehicle data from BQ
        rows = ( p | 'read data' >> beam.io.Read(beam.io.BigQuerySource(query=allData, use_standard_sql=True)))
        
        # Step 2: Defines the key for the data to be the Hardware ID (or vehicle identifier)
        rows_keyed = ( rows | 'KeyByHardwareId' >> beam.Map(lambda x: (x['HardwareId'], x)))
                
        # Step 3: Groups all of the incoming rows by the key (the Hardware ID)
        rows_g = ( rows_keyed | 'group rows' >> beam.GroupByKey())
                     
        # Step 4: Creates all output images (as a base64-encoded image)
        images_out = (rows_g | 'create output images' >> beam.ParDo(GenerateImage(), dfFilteredPlaces))

        # Step 5: Writes all images and metadata back out to BigQuery
        output = ( images_out | 'Write images' >> beam.io.WriteToBigQuery(known_args.output_images,
                                                               schema = 'VehicleType:STRING,WeightClass:STRING,Industry:STRING,Carrier:STRING,GovVocation:STRING,PlaceTypeImage:STRING,PlaceSubTypeImage:STRING,NumPlaceTypeElements:INTEGER,NumPlaceSubTypeElements:INTEGER',
                                                               create_disposition = beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                                                               write_disposition = beam.io.BigQueryDisposition.WRITE_APPEND
                                                              ))
        
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()