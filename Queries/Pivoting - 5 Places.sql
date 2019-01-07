select 
  StopLatitude,
  StopLongitude,
  MAX(case when rn=1 then PlaceLatitude else null end) as Place1Latitude,
  MAX(case when rn=1 then PlaceLongitude else null end) as Place1Longitude,
  MAX(case when rn=1 then PlaceTypeKey else null end) as Place1TypeKey,
  MAX(case when rn=1 then PlaceTypeValue else null end) as Place1TypeValue,
  MAX(case when rn=1 then Distance else null end) as Distance1,
  MAX(case when rn=2 then PlaceLatitude else null end) as Place2Latitude,
  MAX(case when rn=2 then PlaceLongitude else null end) as Place2Longitude,
  MAX(case when rn=2 then PlaceTypeKey else null end) as Place2TypeKey,
  MAX(case when rn=2 then PlaceTypeValue else null end) as Place2TypeValue,
  MAX(case when rn=2 then Distance else null end) as Distance2,
  MAX(case when rn=3 then PlaceLatitude else null end) as Place3Latitude,
  MAX(case when rn=3 then PlaceLongitude else null end) as Place3Longitude,
  MAX(case when rn=3 then PlaceTypeKey else null end) as Place3TypeKey,
  MAX(case when rn=3 then PlaceTypeValue else null end) as Place3TypeValue,
  MAX(case when rn=3 then Distance else null end) as Distance3,
  MAX(case when rn=4 then PlaceLatitude else null end) as Place4Latitude,
  MAX(case when rn=4 then PlaceLongitude else null end) as Place4Longitude,
  MAX(case when rn=4 then PlaceTypeKey else null end) as Place4TypeKey,
  MAX(case when rn=4 then PlaceTypeValue else null end) as Place4TypeValue,
  MAX(case when rn=4 then Distance else null end) as Distance4,
  MAX(case when rn=5 then PlaceLatitude else null end) as Place5Latitude,
  MAX(case when rn=5 then PlaceLongitude else null end) as Place5Longitude,
  MAX(case when rn=5 then PlaceTypeKey else null end) as Place5TypeKey,
  MAX(case when rn=5 then PlaceTypeValue else null end) as Place5TypeValue,
  MAX(case when rn=5 then Distance else null end) as Distance5
FROM
(
  SELECT 
    *,
    ROW_NUMBER() OVER (partition by concat(cast(StopLatitude as string),cast(StopLongitude as string)) ORDER BY Distance ASC) as rn
  FROM 
    `geotab-bi.Project_IndustryClassification.StopLocationTypeDistance` 
  WHERE PlaceTypeValue NOT IN ('tower','pole','level_crossing','fire_hydrant','yes','bicycle_parking','gate','stop_position','switch','toilets','buffer_stop','atm','bicycle_rental','mast','bench','fountain','drinking_water','charging_station','generator','waste_basket','flagpole','telephone','entrance','vending_machine','subway_entrance','phone','recycling','picnic_site','picnic_table','switch','utility_pole','compressed_air','gasometer','viewpoint','chimney','works','parking_space','survey_point','parking_position','abandoned_station','pitch','cable_distribution_cabinet','defibrillator','basketball','turntable','monitoring_station','signal','bbq','stop','siren','railway_crossing','ticket','shower','beacon','holding_position','stop_area','roof','whirlpool','pole','halt','junction','public_bookcase','dog_park','antenna','convenience;car_repair','chemist;stationery;convenience','shed','metal_construction','Massachusetts Trial Court Law Libraries','Portable Toilet Supplier;Trailer Rental Service;Septic System Service;Construction Equipment Supplier;Fence Contractor','yes','communications_dish','Station 2','compensator','culvert','firepit','capacitor_bank','street_cabinet','yes','railway','fire_extinguisher','derail','common','Cape Cod Hospital','terrace','Varnum Brook Elementary School','Middlesex Community College','milestone','monitoring_station=water_quality','yes','proposed','grate','school (historic)','Squannacook Elementary School','hospital (historic)','Orange','aaa','telescope','yes','marker','pylon','Station 3','showers','advertising','tap','Monument','ice_self_vending','bike_rack','sauna')
)
WHERE rn <=5
GROUP BY 
  StopLatitude,
  StopLongitude
