/*

Hardware ID to Industry Mapping View 
This query takes 3rd party Hardware info and joins to industry fleet classification type
*/


WITH OneVinIds AS
  (
   SELECT 
      hardwareid 
    FROM
      (SELECT 
        hardwareid, 
        min(datefrom) as MinDateFrom, 
        max(dateto) as MaxDateTo 
      FROM `geotab-bigdata.Vin.VinHistory`  
      GROUP BY hardwareid 
      HAVING max(vin) =min(vin))
    WHERE MinDateFrom <='2017-12-01' AND (MaxDateTo >='2018-11-01' OR MaxDateTo is null)
  ),
HardwareDb AS 
(
  SELECT 
    O.HardwareId,
    lower(Max(C.DatabaseName)) as DatabaseName
  FROM 
    `geotab-gateway.StoreForward.VehicleClient` V 
  INNER JOIN 
    OneVinIds O ON V.VehicleId=O.hardwareid
  INNER JOIN
    `geotab-gateway.StoreForward.ClientInfo_20181130` C ON V.ClientGuid=C.Guid
  GROUP BY
    O.HardwareId
  HAVING 
    max(C.DatabaseName) = min(C.DatabaseName)
)

SELECT 
  HardwareId,
  H.DatabaseName,
  IHS_CompanyVocation, 
  IHS_CarrierType, 
  IHS_GovVocation
FROM
  HardwareDb H
INNER JOIN
  `geotab-bi.Project_IndustryClassification.IHS_Classification` I ON lower(I.DatabaseName)=H.DatabaseName
WHERE I.IHS_CompanyVocation <> 'UNCLASSIFIED'
