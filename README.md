# H1B-Disclosure-Dataset

1) Data Background

In the Data Mining class, we had the opportunity to analyze data by performing data mining algorithms to a dataset. Our dataset is from Office of Foreign Labor Certification (OFLC). OFLC is a division of the U.S. Department of Labor. The main duty of OFLC is to assist the Secretary of Labor to enforce part of the Immigration and Nationality Act (INA), which requires certain labor conditions exist before employers can hire foreign workers. 
H-1B is a visa category in the United States of America under the INA, section 101(a)(15)(H) which allows U.S. employers to employ foreign workers. The first step employer must take to hire a foreign worker is to file the Labor Condition Application. In this project, we will analyze the data from the Labor Condition Application.

1.1) Introduction to H1B Dataset

The H-1B Dataset selected for this project contains data from employer’s Labor Condition Application and the case certification determinations processed by the Office of Foreign Labor Certification (OFLC) where the date of the determination was issues on or after October 1, 2016 and on or before June 30, 2017.

The Labor Condition Application (LCA) is a document that a perspective H-1B employer files with U.S. Department of Labor Employment and Training Administration (DOLETA) when it seeks to employ non-immigrant workers at a specific job occupation in an area of intended employment for not more than three years.

1.2) Goal of the Project

Our goal for this project is to predict the case status of an application submitted by the employer to hire non-immigrant workers under the H-1B visa program. Employer can hire non-immigrant workers only after their LCA petition is approved. The approved LCA petition is then submitted as part of the Petition for a Non-immigrant Worker application for work authorizations for H-1B visa status.

We want to uncover insights that can help employers understand the process of getting their LCA approved. We will use WEKA software to run data mining algorithms to understand the relationship between attributes and the target variable.

2)Dataset Information:

a) Source: Office of Foreign Labor Certification, U.S. Department of Labor Employment and Training Administration 
b) List Link: https://www.foreignlaborcert.doleta.gov/performancedata.cfm 
c) Dataset Type: Record – Transaction Data 
d) Number of Attributes: 40 
e) Number of Instances: 528,147 
f) Date Created: July 2017

3) Attribute List:

The detailed description of each attribute below is given in the Record Layout file available in the zip folder H1B Disclosure Dataset Files.

The H-1B dataset from OFLC contained 40 attributes and 528,147 instances. The attributes are in the table below. The attributes highlighted bold were removed during the data cleaning process.

1) CASENUMBER 
2)CASESUBMITTED 
3)DECISIONDATE 4)VISACLASS 
5)EMPLOYMENTSTARTDATE 
6)EMPLOYMENTENDDATE 
7)EMPLOYERNAME 8)EMPLOYERADDRESS 
9)EMPLOYERCITY 
10)EMPLOYERSTATE 
11)EMPLOYERPOSTALCODE 
12)EMPLOYERCOUNTRY 13)EMPLOYERPROVINCE 
14)EMPLOYERPHONE 
15)EMPLOYERPHONEEXT 
16)AGENTATTORNEYNAME 
17)AGENTATTORNEYCITY 
18)AGENTATTORNEYSTATE 
19)JOBTITLE 
20)SOCCODE 
21)SOCNAME 
22)NAICSCODE 23)TOTALWORKERS 
24)FULLTIMEPOSITION 
25)PREVAILINGWAGE 26)PWUNITOFPAY 
27)PWSOURCE 28)PWSOURCEYEAR 29)PWSOURCEOTHER 30)WAGERATEOFPAYFROM 31)WAGERATEOFPAYTO 32)WAGEUNITOFPAY 
33)H-1BDEPENDENT 34) WILLFULVIOLATOR 
35) WORKSITECITY 
36)WORKSITECOUNTY 
37)WORKSITESTATE 38)WORKSITEPOSTALCODE 39)ORIGINALCERTDATE 40)CASESTATUS* - __Class Attribute - To be predicted

3.1) Class Attribute

For the H-1B Dataset our class attribute is ‘CASESTATUS’. There are 4 categories of Case Status. The values of CaseStatus attributes are:

1) Certified 
2) Certified_Withdrawn 
3) Withdrawn 
4) Denied

Certified means the LCA of an employer was approved. Certified Withdrawn means the case was withdrawn after it was certified by OFLC. Withdrawn means the case was withdrawn by the employer. Denied means the case was denied OFLC.
