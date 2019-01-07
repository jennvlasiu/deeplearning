/*

IHS Classification View
- Third-party vehicle information
Pre-Processing Query. Purpose of this query is to pull from Geotab Database & 3rd Party (IHS) to gain high level info on customers who have a Geotab telematics device installed.

*/

select
  DatabaseName,
  CustomerName,
  ResellerDescription,
  TotalDevices,
  Region,
  IHS_CompanyName,
  IHS_CompanyVocation,
  IHS_CarrierType,
  IHS_GovVocation,
  IHS_FleetSize,
  IHS_CompanyStat,
  IHS_CompanyCity
FROM
  `geotab-govdata.CustomerClassification.IHS_GovClassification`
UNION ALL
select
  DatabaseName,
  CustomerName,
  ResellerDescription,
  TotalDevices,
  Region,
  IHS_CompanyName,
  IHS_CompanyVocation,
  IHS_CarrierType,
  NULL as IHS_GovVocation,
  IHS_FleetSize,
  IHS_CompanyStat,
  IHS_CompanyCity
FROM
  `geotab-govdata.CustomerClassification.IHS_NonGovClassification`
