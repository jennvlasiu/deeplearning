/* Trips Fully Joined - 5 Places */

select 
  T.*,
  S.* except (StopLatitude, StopLongitude),
  D.* EXCEPT (Vin),
  H.IHS_CompanyVocation,
  H.IHS_CarrierType,
  H.IHS_GovVocation,
  H.DatabaseName
FROM
  `geotab-bigdata.Trips.Trips` T
INNER JOIN 
  `geotab-bi.Project_IndustryClassification.HardwareIndustryMapping` H 
  ON T.HardwareId=H.HardwareId
INNER JOIN 
  -- The hardware IDs being looked at only have a single VIN
  (SELECT Max(Vin) as Vin, HardwareId FROM `geotab-bigdata.Vin.HardwareIdHistory` GROUP BY HardwareId) V 
  ON T.HardwareId=V.HardwareId
INNER JOIN 
  (SELECT Vin, VehicleType, WeightClass FROM `geotab-bigdata.Vin.VinDecode`) D 
  ON V.Vin=D.Vin
LEFT JOIN
  `geotab-bi.Project_IndustryClassification.Stops_FivePlaces` S 
  ON T.StopLatitude=S.StopLatitude AND T.StopLongitude=S.StopLongitude

WHERE 
  T.StartTime >='2017-12-01' and T.StartTime < '2018-12-01'
