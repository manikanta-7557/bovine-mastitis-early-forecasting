#--------------------------------------------------------------
# qPCR Pathogen Pre-processing Script
#--------------------------------------------------------------
# Author: Lirot Hélène
# Date: 2025/02/26
# Description:
# This script processes qPCR data to generate calibration curves 
# for positive controls (1) and Staphylococcus strains (2). 
# It calculates the calibration coefficients (slope and intercept) 
# using linear regression, applies this calibration factor to the qPCR data (3), 
# and calculates the quantity proxy based on the sample mass (4).
#--------------------------------------------------------------

#--------------------------------------------------------------
# Loading the necessary libraries
#--------------------------------------------------------------
library(dplyr)  # Data manipulation
library(ggplot2) # Creating plots

#--------------------------------------------------------------
# Data Import and Preprocessing
#--------------------------------------------------------------
### Read qPCR for calibration range
qPCR_range <- read.csv2("qPCR_pathogen_range.csv")
qPCR_range$CT_value <- as.numeric(qPCR_range$CT_value)

### Read qPCR pathogen data from samples
qPCR_pathogen_quantified <- read.csv2("qPCR_pathogen.csv")
qPCR_pathogen_quantified <- qPCR_pathogen_quantified %>%
  filter(Type_sample == "Sample")

### Read metadata
metadata <- read.csv2("Metadata.csv")


#--------------------------------------------------------------
# 1. Calibration Curve for Positive Controls
#--------------------------------------------------------------
### Filter and visualize calibration curve for positive control samples
positive_control_data <- qPCR_range %>% 
  filter(Sample == "Positive_control")
ggplot(positive_control_data, aes(x = Dilution_number, y = CT_value, color = Select_calibration)) +
  geom_point() +
  facet_wrap(~Pathogen) +
  theme_bw()
rm(positive_control_data)

### Select relevant calibration data
calibration_data <- qPCR_range %>% 
  filter(Sample == "Positive_control" & Select_calibration == "Yes")

# Print dilution type for validation
print(unique(calibration_data$Dilution_type))  # Dilution factor of 5

### Compute calibration coefficients for each pathogen
calibration_coeffs <- data.frame(Pathogen = character(), Slope = numeric(), Intercept = numeric(), stringsAsFactors = FALSE)
pathogen_list_calibration <- unique(calibration_data$Pathogen)
for (p in pathogen_list_calibration) {
  pathogen_data <- calibration_data %>%
    filter(Pathogen == p) %>%
    arrange(Dilution_number)
  max_dilution <- max(pathogen_data$Dilution_number)
  pathogen_data$Nb_copies <- 10 * (5 ^ (max_dilution - pathogen_data$Dilution_number))
  model <- lm(CT_value ~ log10(Nb_copies), data = pathogen_data)
  calibration_coeffs <- rbind(calibration_coeffs, 
                              data.frame(Pathogen = p, 
                                         Slope = round(as.numeric(coef(model)[2]), 2), 
                                         Intercept = round(as.numeric(coef(model)[1]), 2)))
}
rownames(calibration_coeffs) <- NULL
rm(model, p, max_dilution, calibration_data, pathogen_list_calibration, pathogen_data)

#--------------------------------------------------------------
# 2. Calibration Curve for Staphylococcus Strains
#--------------------------------------------------------------
### Filter and visualize calibration curve for Staphylococcus Strains 
staph_data <- qPCR_range %>% filter(Pathogen %in% c("Staph_aureus", "Staph_sp")) %>% filter(Sample %in% c("Staph_aureus", "Staph_xylosus"))
ggplot(staph_data, aes(x = Dilution_number, y = CT_value, color = Select_calibration)) +
  geom_point() +
  facet_wrap(~Pathogen + Sample) +
  theme_bw()
rm(staph_data)

#########
### Staphylococcus aureus
#########
# Initial Concentration:
# Calculation based on Avogadro's number:
# - S. aureus concentration: 290 ng/µL
# - Amplicon length: 2,843,075 bp (Larner-Svensson et al., 2013)
# - Estimated copies: 9.3 × 10^7 copies/µL
concentration_ini_Staph_aureus <- 9.3 * 10^7

### Select relevant calibration data
calibration_data_staph_aureus <- qPCR_range %>% 
  filter(Select_calibration == "Yes", Pathogen == "Staph_aureus", Sample == "Staph_aureus") 

### Print dilution type for validation
print(unique(calibration_data_staph_aureus$Dilution_type)) # Dilution factor of 10

### Compute calibration coefficients
calibration_data_staph_aureus$Nb_copies <- concentration_ini_Staph_aureus / 10^calibration_data_staph_aureus$Dilution_number
model_staph_aureus <- lm(CT_value ~ log10(Nb_copies), data = calibration_data_staph_aureus)
calibration_coeffs <- rbind(calibration_coeffs, 
                            data.frame(Pathogen = "Detection_Staph_aureus_Strain_Staph_aureus", 
                                       Slope = round(as.numeric(coef(model_staph_aureus)[2]), 2), 
                                       Intercept = round(as.numeric(coef(model_staph_aureus)[1]), 2)))
rm(model_staph_aureus, concentration_ini_Staph_aureus, calibration_data_staph_aureus)

#########
### Staphylococcus xylosus 
#########
# Initial Concentration:
# Calculation based on Avogadro's number:
# - S. xylosus concentration: 340 ng/µL
# - Amplicon length: 2,825,246 bp (Heo et al., 2021)
# - Estimated copies: 1.1 × 10^8 copies/µL
concentration_ini_Staph_sp <- 1.1 * 10^8

### Select relevant calibration data
calibration_data_staph_sp <- qPCR_range %>% 
  filter(Select_calibration == "Yes", Pathogen == "Staph_sp", Sample == "Staph_xylosus")

# Print dilution type for validation 
print(unique(calibration_data_staph_sp$Dilution_type))   # Dilution factor of 10

### Compute calibration coefficients
calibration_data_staph_sp$Nb_copies <- concentration_ini_Staph_sp / 10^calibration_data_staph_sp$Dilution_number
model_staph_sp <- lm(CT_value ~ log10(Nb_copies), data = calibration_data_staph_sp)
calibration_coeffs <- rbind(calibration_coeffs, 
                            data.frame(Pathogen = "Detection_Staph_sp_Strain_Staph_xylosus", 
                                       Slope = round(as.numeric(coef(model_staph_sp)[2]), 2), 
                                       Intercept = round(as.numeric(coef(model_staph_sp)[1]), 2)))
rm(model_staph_sp, concentration_ini_Staph_sp, calibration_data_staph_sp)


#--------------------------------------------------------------
# 3. Modification of Pathogen Data for Samples Based on Previous Calibration Curves
#--------------------------------------------------------------
### List of pathogens (without Staphylococcus) to process
pathogen_list_without_staph <- c("Beta_lacta","C_bovis","E_coli","Entero_sp","Klebsie_sp","Myco_bovis","Myco_sp","Proto_sp","S_marces",
                "Str_agala","Str_dysgala","Str_uberis","T_pyo_P_indo","Yeasts")

### Loop through each pathogen in the list
for (p in pathogen_list_without_staph) {
  calibration_coeffs_p <- calibration_coeffs %>% filter(Pathogen == p)
  Slope <- calibration_coeffs_p$Slope  # Extract the slope
  Intercept <- calibration_coeffs_p$Intercept  # Extract the intercept
  
  copie <- ((10 ^ ((select(qPCR_pathogen_quantified, paste(p, "CT_value", sep = "_")) - Intercept) / Slope)))
  colnames(copie) <- paste(p, "copy", sep = "_")

  qPCR_pathogen_quantified <- cbind(qPCR_pathogen_quantified, copie)

  is.na(qPCR_pathogen_quantified[[paste(p, "copy", sep = "_")]]) <- 0
  qPCR_pathogen_quantified[[paste(p, "copy", sep = "_")]] <- as.numeric(
    ifelse(qPCR_pathogen_quantified[[paste(p, "quantity", sep = "_")]] == "-", "0", 
           qPCR_pathogen_quantified[[paste(p, "copy", sep = "_")]]))
}
rm(Slope) ; rm(Intercept) ; rm(copie) ; rm(calibration_coeffs_p)


#########
### Special Cases: Staphylococcus aureus 
#########
### Use the calibration curve created with the Staph aureus strain and detection of Staph aureus.
calibration_coeffs_staph_aureus <- calibration_coeffs %>% 
  filter(Pathogen == "Detection_Staph_aureus_Strain_Staph_aureus")
qPCR_pathogen_quantified$Staph_aureus_copy <- 10^((qPCR_pathogen_quantified$Staph_aureus_CT_value - calibration_coeffs_staph_aureus$Intercept) / calibration_coeffs_staph_aureus$Slope)
qPCR_pathogen_quantified$Staph_aureus_copy[is.na(qPCR_pathogen_quantified$Staph_aureus_copy)] <- 0
rm(calibration_coeffs_staph_aureus)

#########
### Special Cases: Staphylococcus xylosus 
########
### Determine the NAS (Non-aureus staphylococcus Species) CT value based on the Staph aureus and Staph sp CT values
qPCR_pathogen_quantified$NAS_CT_value <- ifelse(
  is.na(qPCR_pathogen_quantified$Staph_sp_CT_value) & is.na(qPCR_pathogen_quantified$Staph_aureus_CT_value),  # If both are missing
  41, # Corresponds to the qPCR detection limit
  ifelse(
    qPCR_pathogen_quantified$Staph_sp_CT_value < 41 & is.na(qPCR_pathogen_quantified$Staph_aureus_CT_value),  # If Staph sp CT < 41 and Staph aureus CT is missing
    qPCR_pathogen_quantified$Staph_sp_CT_value,  
    ifelse(
      is.na(qPCR_pathogen_quantified$Staph_sp_CT_value) & qPCR_pathogen_quantified$Staph_aureus_CT_value < 41,  # If Staph sp CT is missing and Staph aureus CT < 41
      41,  
      ifelse(
        qPCR_pathogen_quantified$Staph_sp_CT_value < qPCR_pathogen_quantified$Staph_aureus_CT_value,  # If Staph sp CT < Staph aureus CT
        (log(2^(-(qPCR_pathogen_quantified$Staph_sp_CT_value - qPCR_pathogen_quantified$Staph_aureus_CT_value)) - 1, 10) * -3.10 + qPCR_pathogen_quantified$Staph_aureus_CT_value),  
        ifelse(
          qPCR_pathogen_quantified$Staph_sp_CT_value == qPCR_pathogen_quantified$Staph_aureus_CT_value,  # If both CT values are equal
          41,  
          41 )))))
qPCR_pathogen_quantified$NAS_CT_value <- as.numeric(qPCR_pathogen_quantified$NAS_CT_value)

### Use the calibration curve created with the Staph xylosus strain and detection of Staph sp to modify the NAS CT_value 
calibration_coeffs_NAS <- calibration_coeffs %>% filter(Pathogen == "Detection_Staph_sp_Strain_Staph_xylosus")
qPCR_pathogen_quantified$NAS_copy <- 10^((qPCR_pathogen_quantified$NAS_CT_value - calibration_coeffs_NAS$Intercept) / calibration_coeffs_NAS$Slope)
qPCR_pathogen_quantified$NAS_copy <- as.numeric(ifelse(qPCR_pathogen_quantified$NAS_CT_value == 41, "0", qPCR_pathogen_quantified$NAS_copy)) # For NAS  treat "CT_value == 41" as a special case and assign 0
rm(calibration_coeffs_NAS)


#--------------------------------------------------------------
# 4. Normalize Copy Numbers Based on Sample Mass
#--------------------------------------------------------------
# Merge metadata with qPCR pathogen data
qPCR_pathogen_quantified <- metadata %>%
  filter(Sampling_status == "Yes")%>%
  select(Lab_ID, Matrices) %>%
  left_join(qPCR_pathogen_quantified, by = "Lab_ID")

# Calculation of the number of copies per sample as a function of the matrices 
pathogen_list <- c("Beta_lacta", "C_bovis", "E_coli", "Entero_sp", "Klebsie_sp", "Myco_bovis", "Myco_sp", "Proto_sp", "S_marces", "Staph_aureus",
                "Str_agala", "Str_dysgala", "Str_uberis", "T_pyo_P_indo", "Yeasts", "NAS")
for (p in pathogen_list) {
  qPCR_pathogen_quantified[[paste(p, "copy_sample", sep = "_")]] <- NA
  qPCR_pathogen_quantified[[paste0(p, "_copy_sample")]] <- 
    ifelse(qPCR_pathogen_quantified$Matrices %in% c("Feces", "Bedding"), 
           (qPCR_pathogen_quantified[[paste0(p, "_copy")]] * 100 / 150) * 1000, 
           ifelse(qPCR_pathogen_quantified$Matrices %in% c("Milk", "Milk_filter", "Tank_milk"), 
                  (qPCR_pathogen_quantified[[paste0(p, "_copy")]] * 100 / 250) * 1000, NA))
}

str(qPCR_pathogen_quantified)  

# write.csv2(qPCR_pathogen_quantified, "qPCR_pathogen_quantified.csv")

