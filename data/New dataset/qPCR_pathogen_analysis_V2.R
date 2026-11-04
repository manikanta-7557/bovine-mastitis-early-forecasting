#--------------------------------------------------------------
# qPCR Pathogen Analysis Script - V2
#--------------------------------------------------------------
# Author: Lirot Hélène
# Date: 2025/07/03
# Description:
# (1) Comparison of the "matrices" (milk, milk filter, and tank milk)
# (2) Principal Component Analysis (PCA) on pathogen data across all matrices
# (3) Clustering of pathogen data from milk samples
#     a. Clustering process
#     b. Clustering description
#     c. Definition of states
#     d. Clustering transitions
#     e. Effect of treatment on transitions
#     f. Effect of somatic cells on transitions
#     g. Clustering robustness
#     h. Duration of infection by pathogens in milk samples
# Each part (1-2-3) can be used independently
#--------------------------------------------------------------

#--------------------------------------------------------------
# Loading the necessary libraries
#--------------------------------------------------------------
# Visualization libraries
library(ggplot2)        # Creating plots
library(ggpubr)         # Arranging ggplot2 visualizations (e.g., ggarrange)
library(ggbeeswarm)     # Visualization of point distributions
library(RColorBrewer)   # Color palettes
library(corrplot)       # Correlation matrix visualization
library(ggh4x)          # Extensions for ggplot2
library(cowplot)        # Plot themes and combining plots
library(grid)           # Grid graphics system (low-level)
library(ggplotify)      # Convert base plots to ggplot objects
library(scales)         # Scaling functions for ggplot2
library(gridExtra)
library(viridis)


# Data manipulation and wrangling
library(dplyr)          # Data manipulation (filter, mutate, summarize, etc.)
library(tidyr)          # Tidying data (pivot_longer, pivot_wider, etc.)
library(stringr)        # String manipulation
library(forcats)        # Handling categorical variables (factors, ordering)
library(reshape2)       # Reshaping data (melt, dcast)

# Statistical analysis
library(FactoMineR)     # Multivariate analysis (PCA, MCA, etc.)
library(factoextra)     # Visualization of multivariate analyses
library(cluster)        # Clustering methods
library(markovchain)    # Markov chain modeling and analysis
library(gtsummary)      # Summary tables for statistical analyses

#--------------------------------------------------------------
# data Import and Preprocessing
#--------------------------------------------------------------

### Read qPCR pathogen data modified by qPCR_pathogen_preprocessing.R
qPCR_pathogen_quantified <- read.csv2("qPCR_pathogen_quantified.csv")

### Read metadata
metadata <- read.csv2("Metadata_V2.csv")
metadata <- metadata %>%
  filter(Sampling_status == "Yes")

### SCC interpolation 
metadata_milk <- metadata%>%
  filter(Matrices =="Milk")%>%
  mutate(Somatic_cells =as.numeric(Somatic_cells),
         Date = as.Date(Date, format = "%d/%m/%Y"))

list_cow<-unique(metadata_milk$Cow)
result <- NULL
for (i in list_cow) {
  data_i<-metadata_milk[metadata_milk$Cow==i,]
  data_i<-data_i[order(data_i[,1],decreasing = F),]
  data_i_cell<-approx(x=data_i$Date,y=log(data_i$Somatic_cells),method = "linear",xout = data_i$Date)
  data_i_cell$Somatic_cells_interpolation <- exp(data_i_cell$y)
  data_i_cell$x<-as.Date(data_i_cell$x)
  data_i_merge<-merge(data_i,data_i_cell,by.x = "Date",by.y = "x")
  
  if(is.null(result)){
    result <- as.data.frame(data_i_merge)
  }else{
    result <- rbind(result, as.data.frame(data_i_merge))
  }
} 
result_2 <- result%>%
  select(Lab_ID, Somatic_cells_interpolation)

metadata_all <- merge(metadata, result_2, all=T)
rm(data_i, data_i_merge, data_i_cell,metadata,metadata_milk,result,result_2)

### Merge qPCR data with metadata
data_merge <- merge(qPCR_pathogen_quantified, metadata_all, by = c("Lab_ID", "Matrices"))

data_merge <- data_merge %>%
  mutate(across(ends_with("copy_sample"), ~ log(. + 1), .names = "{.col}_log"),
         qPCR_16Sech = as.numeric(qPCR_16Sech),
         Pathogen_number = rowSums(select(., ends_with("copy_sample")) != 0))%>%
  mutate(Date = as.Date(Date, format = "%d/%m/%Y"),
         Cow = as.factor(Cow),
         Lab_ID = as.factor(Lab_ID),
         Farm = as.factor(Farm),
         Matrices = as.factor(Matrices))
row.names(data_merge) <- data_merge$Lab_ID


### Addition of missing data to obtain a homogeneous spread between samples
missing_value <- read.csv2("Missing_sampling.csv")

missing_value <- missing_value %>%
  mutate(Lab_ID = NA, 
         Date = as.Date(Date, format = "%d/%m/%Y"),  
         Cow = as.factor(Cow),
         Farm =as.factor(Farm), 
         Matrices = 'Milk')

data <- bind_rows(data_merge, missing_value)





#--------------------------------------------------------------
# 1.Comparison of matrices
#--------------------------------------------------------------

### Filter and clean all-related data
data_all <- data %>%
  select(ends_with("copy_sample"),-Beta_lacta_copy_sample, Matrices,Pathogen_number) %>%
  filter(Matrices %in% c("Milk_filter", "Milk",  "Bedding", "Feces", "Tank_milk")) 

### Convert to long format
data_all_long <- data_all %>%
  pivot_longer(cols = ends_with("copy_sample"), 
               names_to = "Pathogen", 
               values_to = "Copy")

### Presence-absence variable
data_all_long <- data_all_long %>%
  mutate(Presence_absence = ifelse(Copy == 0, 0, 1))

### Data Modification for Percentage Calculation
data_all_percentage <- data_all_long %>%
  mutate(Matrices = as.factor(Matrices), Pathogen = as.factor(Pathogen)) %>%
  group_by(Matrices, Pathogen, Presence_absence) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(Matrices, Pathogen) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  filter(Presence_absence == 1) %>%
  ungroup() 

### Transformation of Pathogen Variable with Expressions
data_all_percentage <- data_all_percentage %>%
  mutate(Pathogen_name = case_when(
    Pathogen == 'C_bovis_copy_sample' ~ 'italic("C. bovis")',
    Pathogen == 'S_marces_copy_sample' ~ 'italic("Serratia marcescens")',
    Pathogen == 'Str_agala_copy_sample' ~ 'italic("S. agalactiae")',
    Pathogen == 'Str_uberis_copy_sample' ~ 'italic("S. uberis")',
    Pathogen == 'Staph_aureus_copy_sample' ~ 'italic("S. aureus")',
    Pathogen == 'Entero_sp_copy_sample' ~ 'italic("Enterococcus")~" spp."',
    Pathogen == 'Klebsie_sp_copy_sample' ~ 'italic("Klebsiella")~" spp."',
    Pathogen == 'Myco_bovis_copy_sample' ~ 'italic("M. bovis")',
    Pathogen == 'Myco_sp_copy_sample' ~ 'italic("Mycoplasma")~" spp."',
    Pathogen == 'Proto_sp_copy_sample' ~ 'italic("Prototheca")~" spp."',
    Pathogen == 'Str_dysgala_copy_sample' ~ 'italic("S. dysgalactiae")',
    Pathogen == 'E_coli_copy_sample' ~ 'italic("E. coli")',
    Pathogen == 'NAS_copy_sample' ~ 'NAS',
    Pathogen == 'T_pyo_P_indo_copy_sample' ~ 'italic("T. pyogenes/P. indolicus")',
    Pathogen == 'Yeasts_copy_sample' ~ 'Yeasts'))
data_all_percentage$Matrices <- factor(data_all_percentage$Matrices, levels = c("Milk", "Milk_filter","Tank_milk",  "Bedding", "Feces"))
data_all_percentage$Pathogen_name <- factor(data_all_percentage$Pathogen_name,
                                            levels = c('italic("E. coli")',
                                                       'NAS',
                                                       'italic("S. aureus")',
                                                       'italic("C. bovis")',
                                                       'italic("Mycoplasma")~" spp."',
                                                       'italic("T. pyogenes/P. indolicus")',
                                                       'italic("Enterococcus")~" spp."',
                                                       'italic("S. dysgalactiae")',
                                                       'italic("S. agalactiae")',
                                                       'italic("Serratia marcescens")',
                                                       'Yeasts',
                                                       'italic("Prototheca")~" spp."',
                                                       'italic("S. uberis")',
                                                       'italic("Klebsiella")~" spp."'))

plot_matrices <- ggplot(data_all_percentage, aes(y = percentage, x = Matrices, fill = Matrices)) +
  geom_bar(stat = 'identity') +
  facet_wrap(~Pathogen_name, labeller = label_parsed, ncol = 5) +  
  scale_x_discrete(labels = c("Milk_filter" = "Milk filter","Tank_milk" = "Tank milk", "Feces" = "Faeces")) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
  ylab("Percentage of samples with the pathogen") +
  scale_fill_manual(values = c("steelblue3", "snow4","snow3", "gold", "tan4"),
                    labels = c("Milk" = "Milk", "Milk_filter" = "Milk filter",
                               "Tank_milk" = "Tank milk",
                               "Bedding" = "Bedding","Feces" = "Faeces"))+
  theme(
    axis.title = element_text(size = 10),
    axis.text.x = element_text(size = 10, angle = 90, vjust = 0.5),
    strip.text = element_text(size = 8),
    # strip.text.y.left = element_text(angle = 0),
    legend.position = "top",  
    legend.title = element_blank(),  
    legend.text = element_text(size = 10),    
    legend.key.size = unit(0.4, "cm"))

# ggsave("Figure2.pdf", plot = plot_matrices, width = 180, height = 120, units = "mm")


### Number of pathogens per profil
data_all %>%
  select(Matrices, Pathogen_number) %>%
  mutate(Pathogen_number = as.factor(Pathogen_number))%>%
  tbl_summary(Matrices)

ggplot(data_all, aes(x=Matrices, fill=as.factor(Pathogen_number)))+
  geom_bar()

#--------------------------------------------------------------
# 2.PCA on pathogen data (log(x+1)) from all matrices 
#--------------------------------------------------------------

### Prepare the data for PCA analysis
data_pca <- data%>% 
  select(Matrices, Farm, ends_with("_log"), -Beta_lacta_copy_sample_log)%>%
  mutate(Matrices = case_when(
    Matrices == "Milk_filter" ~ "I",
    Matrices == "Milk" ~ "M",
    Matrices == "Tank_milk" ~ "T",
    Matrices == "Feces" ~ "F",
    Matrices == "Bedding" ~ "B"))%>%
  select(where(~ !all(. == 0, na.rm = TRUE)))%>%
  mutate(Farm = factor(Farm),
         Matrices = factor(Matrices),
         Matrices_Farm = as.factor(paste(Matrices, Farm,sep="")))%>%
  select(-Farm)

### Perform PCA
pca_results <- PCA(data_pca, quali.sup = c("Matrices", "Matrices_Farm"), scale.unit = TRUE)

### Plot variance
fviz_eig(pca_results, choice = "variance",addlabels = TRUE, hjust = -0.3)

### Plot Eigenvalues 
fviz_eig(pca_results, choice = "eigenvalue", addlabels = TRUE)

### Correlation plot
rownames(pca_results$var$cos2) <- dplyr::recode(rownames(pca_results$var$cos2),
                                                 "C_bovis_copy_sample_log" = "C. bovis",
                                                 "E_coli_copy_sample_log" = "E. coli",
                                                 "Entero_sp_copy_sample_log" = "Enterococcus spp.",
                                                 "Klebsie_sp_copy_sample_log" = "Klebsiella spp.",
                                                 "Myco_sp_copy_sample_log" = "Mycoplasma spp.",
                                                 "S_marces_copy_sample_log" = "S. marcescens",
                                                 "Proto_sp_copy_sample_log" = "Prototheca spp.",
                                                 "Staph_aureus_copy_sample_log" = "S. aureus",
                                                 "NAS_copy_sample_log" = "Non-aureus Staphylococci",
                                                 "Str_agala_copy_sample_log" = "S. agalactiae",
                                                 "Str_dysgala_copy_sample_log" = "S. dysgalactiae",
                                                 "Str_uberis_copy_sample_log" = "S. uberis",
                                                 "T_pyo_P_indo_copy_sample_log" = "T. pyogenes/P. indolicus",
                                                 "Yeasts_copy_sample_log" = "Yeasts")
corrplot(pca_results$var$cos2,cl.ratio=1, tl.col= 'black')

### Create PCA variable plot
rownames(pca_results$var$coord) <- dplyr::recode(rownames(pca_results$var$coord),
                                                 "C_bovis_copy_sample_log" = "C. bovis",
                                                 "E_coli_copy_sample_log" = "E. coli",
                                                 "Entero_sp_copy_sample_log" = "Enterococcus spp.",
                                                 "Klebsie_sp_copy_sample_log" = "Klebsiella spp.",
                                                 "Myco_sp_copy_sample_log" = "Mycoplasma spp.",
                                                 "S_marces_copy_sample_log" = "S. marcescens",
                                                 "Proto_sp_copy_sample_log" = "Prototheca spp.",
                                                 "Staph_aureus_copy_sample_log" = "S. aureus",
                                                 "NAS_copy_sample_log" = "NAS",
                                                 "Str_agala_copy_sample_log" = "S. agalactiae",
                                                 "Str_dysgala_copy_sample_log" = "S. dysgalactiae",
                                                 "Str_uberis_copy_sample_log" = "S. uberis",
                                                 "T_pyo_P_indo_copy_sample_log" = "T. pyo./P. indo.",
                                                 "Yeasts_copy_sample_log" = "Yeasts")

#########
# Axes 1-2
# #########

pca_var_plot_1_2 <- plot(pca_results, select = "cos2 0.25", invisible = "quali",
                         choix="var", label = "var",
                         axes = c(1,2),
                         title = "", cex=0.4)

pca_ind_plot_1_2 <- plot.PCA(pca_results, choix = "ind", habillage = 1, label = "quali", 
                             axes = c(1, 2), 
                             col.hab = c( "gold" ,  "tan3", "grey","steelblue2","darkgreen"),
                             col.quali = "black", shadowtext = T, cex = 0.5,
                             repel = TRUE, point.shape=20, title = "")

pca_plot_1_2 <- ggarrange(pca_ind_plot_1_2, pca_var_plot_1_2,   
                          labels = c('a)', 'b)'),
                          align='v')
# ggsave("Figure2.pdf", plot = pca_plot_1_2 , width = 170, height = 100, units = "mm" ) 


#########
# Axes 3-4
#########
pca_var_plot_3_4 <- plot.PCA(pca_results, choix="var", label = "var",
                             axes = c(3,4),
                             title = "", cex=0.6)
pca_ind_plot_3_4 <- plot.PCA(pca_results, choix = "ind", habillage = 1, label = "quali", 
                             axes = c(3,4), 
                             col.hab = c( "gold" ,  "tan3", "grey","steelblue2","darkgreen"),
                             col.quali = "black", shadowtext = T, cex = 0.6,
                             repel = TRUE, point.shape=20, title = "")
pca_plot_3_4 <- ggarrange(pca_ind_plot_3_4,pca_var_plot_3_4,   
                          common.legend = TRUE, legend = "right", 
                          labels = c('a)', 'b)'),
                          align='v')
# ggsave("Figure2_bis", plot = pca_plot_3_4 , width = 170, height = 100, units = "mm" ) 
rm(pca_results, pca_var_plot_1_2, pca_var_plot_3_4, pca_ind_plot_1_2, pca_ind_plot_3_4, pca_plot_1_2, pca_plot_3_4, data_pca)


#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   a. Clustering process
#--------------------------------------------------------------

### Filter the data for Milk matrices
milk_data <- data %>%
  filter(Matrices == "Milk") 

milk_data_matrix <- milk_data %>%
  filter(Lab_ID != 'NA')%>%
  select(ends_with("_log"),-Beta_lacta_copy_sample_log) %>%
  data.matrix()


### Compute Euclidean distance matrix
euclidean_dist <- dist(milk_data_matrix, method = "euclidean")

### Perform classical multidimensional scaling (MDS) for dimension reduction
mds <- cmdscale(euclidean_dist, k = 15, eig = TRUE)  
explained_variance <- round((mds$eig / sum(mds$eig)) * 1000) / 1000
cumulative_variance <- cumsum(explained_variance)
plot(1:10, explained_variance[1:10], ylim = c(0, 0.5), type = "o", lwd = 2,
     main = "Variance Explained (Milk Compartment)", xlab = "Number of Dimensions", ylab = "Percentage of Variance")
abline(h = 1/10, lty = 1, col = 2, lwd = 2)
rm(mds)


### Determine the optimal number of clusters 
# Method 1: Within-cluster sum of squares (WSS)
wss_plot <- fviz_nbclust(milk_data_matrix, FUN = hcut, method = "wss", linecolor = "darkblue") +
  labs(title = "WSS")
# Method 2: Silhouette score
silhouette_plot <- fviz_nbclust(milk_data_matrix, FUN = hcut, method = "silhouette", linecolor = "darkblue") +
  labs(title = "Silhouette")
# Method 3: Gap statistic
gap_stat <- clusGap(milk_data_matrix,
                    FUN = function(x, k) hcut(x, k = k, hc_method = "ward.D2"), 
                    K.max = 10, B = 50)  
gap_stat_plot <- fviz_gap_stat(gap_stat, linecolor = "darkblue") +
  labs(title = "Gap Statistic")
cluster_number_plot <- ggarrange(wss_plot, silhouette_plot, gap_stat_plot, ncol = 3, labels = c("a)", "b)", "c)"))
# ggsave("Figure_sup_mat2.pdf", plot = cluster_number_plot, width = 170, height = 100, units = "mm")
rm(wss_plot, silhouette_plot, gap_stat_plot, gap_stat, cumulative_variance, explained_variance, cluster_number_plot)


### Perform hierarchical clustering (Ward's method)
ward_cluster <- hclust(euclidean_dist, method = "ward.D2")
group_assignments <- cutree(ward_cluster, k = 6)  

### Produce a graphic representation
dendrogram_plot <- fviz_dend(ward_cluster, cex = 0.5, lwd = 0.25, k = 6, 
                             k_colors = "Set2", rect = TRUE, rect_fill = TRUE,rect_border = "Set2", 
                             show_labels = FALSE, main = "")

### Define custom colors for the clusters
dendrogram_attributes <- attributes(dendrogram_plot)$dendrogram
tree_order <- order.dendrogram(dendrogram_attributes)
group_assignments <- factor(group_assignments, levels = unique(group_assignments[tree_order]))
color_palette <- brewer.pal(n = 6, name = "Set2")
names(color_palette) <- unique(group_assignments[tree_order])
new_cluster_names <- sapply(names(color_palette), function(colname) {
  switch(colname,
         "1" = "C", 
         "2" = "A", 
         "3" = "B", 
         "4" = "D", 
         "5" = "E", 
         "6" = "F")})
names(color_palette) <- new_cluster_names

### Extract the coordinates for the first 6 dimensions for classical multidimensional scaling 
mds_result <- cmdscale(euclidean_dist, k = 6, eig = TRUE)
mds_coords <- mds_result$points
eig_values <- mds_result$eig

variance_explained <- eig_values / sum(eig_values[eig_values > 0]) * 100
var1 <- round(variance_explained[1], 1)
var2 <- round(variance_explained[2], 1)
var3 <- round(variance_explained[3], 1)
var4 <- round(variance_explained[4], 1)
var5 <- round(variance_explained[5], 1)
var6 <- round(variance_explained[6], 1)

group_coordinates <- data.frame(
  dim1 = mds_coords[, 1],
  dim2 = mds_coords[, 2],
  dim3 = mds_coords[, 3],
  dim4 = mds_coords[, 4],
  dim5 = mds_coords[, 5],
  dim6 = mds_coords[, 6],
  Group = as.factor(group_assignments))
group_coordinates <- group_coordinates %>%
  mutate(Cluster = case_when(
    Group == 1 ~ "C",
    Group == 2 ~ "A",
    Group == 3 ~ "B",
    Group == 4 ~ "D",
    Group == 5 ~ "E",
    Group == 6 ~ "F"))

### Plot: 
# Dimension 1 vs Dimension 2
plot_1_2_milk <- ggplot(group_coordinates, aes(dim1, dim2, color = Cluster)) +
  geom_point(size= 0.4) +
  stat_ellipse(level = 0.95, linewidth = 0.4, type = "norm") +
  scale_color_manual(values = color_palette) +
  guides(color = guide_legend(title = "Cluster", order = 1)) +
  xlab(paste0("Dimension 1 (", var1, "%)")) +
  ylab(paste0("Dimension 2 (", var2, "%)")) +
  theme_minimal()
# Dimension 3 vs Dimension 4
plot_3_4_milk <- ggplot(group_coordinates, aes(dim3, dim4, color = Cluster)) +
  geom_point(size= 0.4) +
  stat_ellipse(level = 0.95, linewidth = 0.4, type = "norm") +
  scale_color_manual(values = color_palette) +
  guides(color = guide_legend(title = "Cluster", order = 1)) +
  xlab(paste0("Dimension 3 (", var3, "%)")) +
  ylab(paste0("Dimension 4 (", var4, "%)")) +
  theme_minimal()

# Dimension 5 vs Dimension 6
plot_5_6_milk <- ggplot(group_coordinates, aes(dim5, dim6, color = Cluster)) +
  geom_point(size= 0.4) +
  stat_ellipse(level = 0.95, linewidth = 0.4, type = "norm") +
  scale_color_manual(values = color_palette) +
  guides(color = guide_legend(title = "Cluster", order = 1)) +
  xlab(paste0("Dimension 5 (", var5, "%)")) +
  ylab(paste0("Dimension 6 (", var6, "%)")) +
  theme_minimal()

plot_cluster_ward <- ggarrange(plot_1_2_milk, plot_3_4_milk, ncol = 2, labels = c("a)", "b)"),
                               common.legend = TRUE, legend = 'right')
# ggsave("Figure_supp_mat3.pdf", plot = plot_cluster_ward, width = 170, height = 100, units = "mm" ) 
rm(plot_cluster_ward, plot_1_2_milk, plot_3_4_milk, plot_5_6_milk, dendrogram_plot,dendrogram_attributes)


#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   b. Clustering description
#--------------------------------------------------------------

### Clustering description of profiles 
milk_data <- milk_data %>%
  mutate(Lab_ID = as.character(Lab_ID))

group_df <- tibble( Lab_ID = names(group_assignments),
                    Groupe = as.factor(group_assignments))

milk_data <- milk_data %>%
  left_join(group_df, by = "Lab_ID") %>%
  mutate(Profiles = case_when(
    Groupe == 1 ~ "C",
    Groupe == 2 ~ "A",
    Groupe == 3 ~ "B",
    Groupe == 4 ~ "D",
    Groupe == 5 ~ "E",
    Groupe == 6 ~ "F",
    TRUE ~ as.character(Groupe)))


group_summary <- milk_data %>%
  group_by(Profiles) %>%
  summarise(count = n(), .groups = "drop")

milk_data <- milk_data %>%
  left_join(group_summary, by = "Profiles") %>%
  mutate(Profile_label = paste0("Profile ", Profiles, " (n=", count, ")"))


milk_data_long <- milk_data %>%
  select(ends_with("copy_sample"), -Beta_lacta_copy_sample, Profiles, Date, Profile_label) %>%
  pivot_longer(cols = ends_with("copy_sample"), 
               names_to = "Pathogen", 
               values_to = "Copy")

Pathogen_name <- c('C_bovis_copy_sample' = expression(italic("C. bovis")),
                    'E_coli_copy_sample' = expression(italic("E. coli")),
                    'Entero_sp_copy_sample' = expression(italic("Enterococcus")~" spp."),
                    'Klebsie_sp_copy_sample' = expression(italic("Klebsiella")~" spp."),
                    'Myco_bovis_copy_sample' = expression(italic("M. bovis")),
                    'Myco_sp_copy_sample' = expression(italic("Mycoplasma")~" spp."),
                    'Proto_sp_copy_sample' = expression(italic("Prototheca")~" spp."),
                    'S_marces_copy_sample' = expression(italic("S. marcescens")),
                    'Staph_aureus_copy_sample' = expression(italic("S. aureus")),
                    'NAS_copy_sample' = expression(italic("Non-aureus Staphylococci")),
                    'Str_agala_copy_sample' = expression(italic("S. agalactiae")),
                    'Str_dysgala_copy_sample' = expression(italic("S. dysgalactiae")),
                    'Str_uberis_copy_sample' = expression(italic("S. uberis")),
                    'T_pyo_P_indo_copy_sample' = expression(italic("T. pyogenes/P. indolicus")),
                    'Yeasts_copy_sample' = "Yeasts")
ordered_cols <- color_palette[order(names(color_palette))]
strip <- strip_themed(background_x = elem_list_rect(fill = ordered_cols))

### Pathogen Distribution Graph - abondance 
milk_data_long <- milk_data_long %>%
  mutate(present = ifelse(Copy > 0, 1, 0))%>%
  filter(Profiles !='NA')

prev_df <- milk_data_long %>%
  group_by(Pathogen, Profiles) %>%
  summarise(prevalence = mean(present) * 100) %>%
  ungroup()

### Plot 1 : Pathogen quantity
plot_pathogen_1 <- ggplot(milk_data_long, aes(x = Pathogen, y = Copy + 1, fill = Profiles)) +
  geom_quasirandom(size = 0.01) +
  theme_minimal() +
  scale_x_discrete(labels = Pathogen_name) +
  scale_y_log10() +
  ylab("Estimated number of copies per mL") +
  theme(
    axis.title.y = element_text(size = 10),
    axis.text.y = element_text(size = 6),
    axis.ticks.y = element_line(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, size = 6),
    legend.position = "none",
    # axis.line.y = element_line(colour = "black"),
    strip.text = element_text(size = 6),
    axis.text = element_text(size = 6),
    plot.margin = margin(10, 40, 10, 10),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.3)) +
  facet_wrap2(~Profile_label, ncol = 2, dir = "h", strip = strip) +
  scale_fill_manual(values = color_palette)

### Plot 2 : Prevalence
plot_pathogen_2 <- ggplot(prev_df, aes(x = Pathogen, y = prevalence, fill = Profiles)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.4) +  
  scale_y_continuous(
    name = "Prevalence (%)",
    limits = c(0, 100),
    sec.axis = dup_axis()
  ) +
  scale_x_discrete(labels = Pathogen_name) +
  theme_minimal() +
  facet_wrap(~ Profiles, ncol = 2) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, size = 6, color= "white"),
    axis.title.x = element_text(size = 10,color = "white"),
    strip.text = element_text(size = 6),
    axis.title.y.right = element_text(size = 10),
    axis.text.y.right = element_text(size = 6),
    axis.ticks.y.right = element_line(),
    # axis.line.y.right = element_line(colour = "black"),
    # axis.line.x = element_line(colour = "black"),
    axis.ticks.y.left = element_blank(),
    axis.text.y.left = element_blank(),
    axis.title.y.left = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.2),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.position = "none",
    plot.margin = margin(10, 13, 10,42)
  ) +
  scale_fill_manual(values = color_palette)

plot_pathogen <- ggdraw() +
  draw_plot(plot_pathogen_2, 0, 0, 1, 1) +  
  draw_plot(plot_pathogen_1, 0, 0, 1, 1)   


### Somatic Cell Count 

plot_somatic_cells <- ggplot(milk_data%>%filter(Profiles !='NA'), aes(x = as.factor(Profiles), y = as.numeric(Somatic_cells_interpolation), fill = Profiles)) +
  geom_boxplot(linewidth = 0.2, outlier.size = 0.5) +
  theme_bw() +
  geom_hline(yintercept = 200, colour = "red", linewidth = 0.8) +
  scale_y_log10() +
  scale_fill_manual(values = color_palette) +
  ylab("Somatic cell count (x1,000/mL)") +
  xlab("Profiles") +
  theme(legend.position = "none")+
  theme(axis.text=element_text(size=6),
        axis.title=element_text(size=8),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.3),
        panel.grid.major.y = element_line(linewidth = 0.2),
        panel.grid.minor.y = element_line(linewidth = 0.2),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())

milk_data%>%
  select(Profiles, Somatic_cells_interpolation)%>%
  mutate(Somatic_cells_interpolation = as.numeric(Somatic_cells_interpolation))%>%
  tbl_summary(by=Profiles)


### Sample Distribution by Farm
plot_farms <- ggplot(milk_data%>%filter(Profiles !='NA'), aes(fill = as.factor(Profiles), x = as.factor(Farm))) +
  geom_bar() +
  theme_bw() +
  scale_y_continuous() +
  scale_fill_manual(values = color_palette) +
  guides(fill = guide_legend(title = "Profiles", order = 1, override.aes = list(size = 4))) +
  xlab("Farms") +
  ylab("Number of samples")+
  theme(axis.text=element_text(size=6),
        axis.title=element_text(size=8),
        legend.text = element_text(size = 6),
        legend.title = element_text(size = 8),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.3),
        panel.grid.major.y = element_line(linewidth = 0.2),
        panel.grid.minor.y = element_line(linewidth = 0.2),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())

### Cluster Description Arrangement
plot_description_cluster <- ggarrange(dendrogram_plot,
                                      ggarrange(plot_pathogen,
                                        ggparagraph(text = " "), 
                                        ggarrange(plot_farms, ggparagraph(text = "   "), plot_somatic_cells, 
                                                  ncol = 1, 
                                                  labels = c("c)", "", "d)"),
                                                 common.legend = TRUE, legend = 'right',
                                                  align = 'hv', heights = c(1, 0., 1)),
                                        ncol = 3, 
                                       labels = "b)",
                                       widths = c(1.30, 0.05, 0.70)),ncol = 1, heights = c(0.50, 1.5),labels = "a)")
ggsave("Figure_4.pdf", plot_description_cluster, width = 170, height = 140, units = "mm" )


### SCC ~ Pathogen ~ Profiles
milk_data_select <- milk_data %>%
  select(Somatic_cells_interpolation, Lab_ID, Groupe, ends_with("copy_sample"), -Beta_lacta_copy_sample) %>%
  pivot_longer(cols = ends_with("copy_sample")) %>%
  mutate(Pathogen = name)%>%
  mutate(value = ifelse(value > 0, 1, 0),
         Somatic_cells_interpolation = as.numeric(Somatic_cells_interpolation))%>%
  mutate(Cluster = case_when(
    Groupe == 1 ~ "C",
    Groupe == 2 ~ "A",
    Groupe == 3 ~ "B",
    Groupe == 4 ~ "D",
    Groupe == 5 ~ "E",
    Groupe == 6 ~ "F"))

milk_data_select_patho <- milk_data_select%>%
  filter(value==1)

Pathogen_name <- c('C_bovis_copy_sample' = expression(italic("C. bovis")),
                   'E_coli_copy_sample' = expression(italic("E. coli")),
                   'Entero_sp_copy_sample' = expression(italic("Enterococcus")~" spp."),
                   'Klebsie_sp_copy_sample' = expression(italic("Klebsiella")~" spp."),
                   'Myco_bovis_copy_sample' = expression(italic("M. bovis")),
                   'Myco_sp_copy_sample' = expression(italic("Mycoplasma")~" spp."),
                   'Proto_sp_copy_sample' = expression(italic("Prototheca")~" spp."),
                   'S_marces_copy_sample' = expression(italic("S. marcescens")),
                   'Staph_aureus_copy_sample' = expression(italic("S. aureus")),
                   'NAS_copy_sample' = expression("NAS"),
                   'Str_agala_copy_sample' = expression(italic("S. agalactiae")),
                   'Str_dysgala_copy_sample' = expression(italic("S. dysgalactiae")),
                   'Str_uberis_copy_sample' = expression(italic("S. uberis")),
                   'T_pyo_P_indo_copy_sample' = expression(italic("T. pyogenes/P. indolicus")),
                   'Yeasts_copy_sample' = "Yeasts")

milk_data_select_patho$Pathogen <- factor(milk_data_select_patho$Pathogen, levels = names(Pathogen_name))


SCC_patho <- ggplot(milk_data_select_patho, aes(y = Somatic_cells_interpolation, x = Pathogen)) +
  geom_quasirandom(aes(color = Cluster), width = 0.3, size = 0.5) +
  scale_y_log10() +
  scale_x_discrete(labels = Pathogen_name) +
  theme_bw() +
  geom_hline(yintercept = 200, colour = "red", linewidth = 0.6) +
  scale_color_manual(values = color_palette) +
  ylab("Somatic cells count (x 1,000 cell/mL)") + 
  labs(color = "Profiles") +
  theme(axis.title.y = element_text(size = 10),
        axis.text.x = element_text(angle = 90, vjust = 0.5),
        strip.text = element_text(size = 7),  
        axis.text = element_text(size = 7),
        panel.grid.major.y = element_line(linewidth = 0.2),
        panel.grid.minor.y = element_line(linewidth = 0.2),
        panel.grid.major.x = element_line(linewidth = 0.2),
        panel.grid.minor.x = element_blank())

# ggsave("Figure_8.pdf", SCC_patho, width = 170, height = 120, units = "mm" )


### Explanation of Somatic Cell Outliers
# Profil C
cell_Group_C <- milk_data %>%
  filter(Profiles == "C") %>%
  select(Farm, Cow, Somatic_cells_interpolation) %>%
  mutate(Somatic_cells_interpolation = as.numeric(Somatic_cells_interpolation))
ggplot(cell_Group_C, aes(x = Cow, y = Somatic_cells_interpolation)) +
  geom_point() +
  geom_hline(yintercept = 200, colour = "red", linewidth = 1.1) +
  scale_y_log10() +
  ggtitle("Profile C - Somatic Cells")

# Profil A
cell_Group_A <- milk_data %>%
  filter(Profiles == "A") %>%
  select(Farm, Cow, Somatic_cells_interpolation) %>%
  mutate(Somatic_cells_interpolation = as.numeric(Somatic_cells_interpolation))
ggplot(cell_Group_A, aes(x = Cow, y = Somatic_cells_interpolation)) +
  geom_point() +
  geom_hline(yintercept = 200, colour = "red", size = 1.1) +
  scale_y_log10() +
  ggtitle("Profile A - Somatic Cells") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

### Group F: Are Yeasts and T. pyogenes/P. indolicus present at the same time?
Patho_group_F <- milk_data %>%
  filter(Profiles == "F") %>%
  select(Farm, Cow, Yeasts_copy_sample, T_pyo_P_indo_copy_sample) %>%
  mutate(Yeast = ifelse(Yeasts_copy_sample == 0, 0, "Yeast"),
         T_pyo_P_indo = ifelse(T_pyo_P_indo_copy_sample == 0, 0, "T. pyo/P. indo"))
Patho_group_F_table <- table(Patho_group_F$Yeast, Patho_group_F$T_pyo_P_indo)
Patho_group_F_table


### S. aureus and somatic cell count
milk_data %>%
  select(Staph_aureus_copy_sample,Somatic_cells_interpolation) %>%
  mutate(Staph_aureus = ifelse(Staph_aureus_copy_sample>0, 1, 0),
         Somatic_cells_interpolation = as.numeric(Somatic_cells_interpolation))%>%
  tbl_summary(by=Staph_aureus)


### Number of pathogens per profil
milk_data %>%
  select(Profiles, Pathogen_number, Farm) %>%
  tbl_summary(Profiles)


### Description of the profiles as a function of the amount of 16S DNA
data_all <- merge(milk_data, data, all = TRUE)
data_total_patho <- data_all %>% 
  select(Lab_ID, Matrices, Profiles, Farm, Cow, ends_with("copy_sample"), qPCR_16Sech) %>%
  filter(Matrices != 'NA')%>%
  mutate(Total_pathogen_load = rowSums(select(., ends_with("copy_sample")), na.rm = TRUE)) %>%
  mutate(qPCR_16Sech = as.numeric(qPCR_16Sech),
         Matrices_label = case_when(
           Matrices == "Tank_milk" ~ "Tank milk",
           Matrices == "Milk_filter" ~ "Milk filter",
           Matrices == "Milk" ~ "Milk",
           Matrices == "Feces" ~ "Feces",
           Matrices == "Bedding" ~ "Bedding"))

figure_16S <- ggplot(data_total_patho, aes(x = Total_pathogen_load + 1, y = qPCR_16Sech + 1, 
                                           color = Profiles, shape = Matrices_label, size = Matrices_label)) +
  geom_point() +
  stat_ellipse(data = subset(data_total_patho, Matrices_label == "Milk"),
               aes(x = Total_pathogen_load + 1, y = qPCR_16Sech + 1, color = Profiles),
               level = 0.95, linewidth = 0.7) +
  scale_y_log10(limits = c(1, 2e11)) +
  scale_x_log10(limits = c(1, 1e+10)) +
  scale_colour_manual(values = color_palette) +  # Use customized color palette
  guides(color = guide_legend(title = "Milk profiles", order = 1)) +  # Ordered legend
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  scale_shape_manual(values = c(4, 17, 19, 0, 5), name = "Matrices") +
  scale_size_manual(values = c(0.7, 0.7, 0.6, 0.7, 0.7), name = "Matrices") +
  theme_bw() +
  guides(
    color = guide_legend(
                         override.aes = list(size = 2)), 
    shape = guide_legend(override.aes = list(size =2)),  
    size = guide_legend(override.aes = list(size = 2)))+
  xlab("Total pathogen quantity detected by the PathoProof kit") + 
  ylab("Total 16S rRNA gene")

# ggsave("Figure_5.pdf", plot = figure_16S , width = 170, height = 120, units = "mm" ) 
# rm(plot_description_cluster, plot_farms, plot_pathogen, plot_somatic_cells, Patho_group_F, Patho_group_F_table,
#    group_summary, data_total_patho, data_all, cell_Group_A, cell_Group_C, figure_16S)


#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   c. States definition
#--------------------------------------------------------------

# Before calculating transitions between groups, it is necessary to define stable states 
milk_data <- milk_data %>%
  mutate(Corrected_profiles = case_when(
    is.na(Profiles) ~ 'No_sample',
    rowSums(across(ends_with("copy_sample") & !starts_with("Beta_lacta"), ~ .x, .names = "selected"), na.rm = TRUE) == 0 ~ "A*",  
    NAS_copy_sample > 0 & C_bovis_copy_sample == 0 & Str_uberis_copy_sample == 0 & 
      Yeasts_copy_sample == 0 & T_pyo_P_indo_copy_sample == 0 ~ "B*",  
    C_bovis_copy_sample > 0 & Str_uberis_copy_sample > 0 ~ "C*",  
    C_bovis_copy_sample > 0 & Str_uberis_copy_sample == 0 ~ "D*",  
    Str_uberis_copy_sample > 0 & C_bovis_copy_sample == 0 ~ "E*",  
    (Yeasts_copy_sample > 0 | T_pyo_P_indo_copy_sample > 0) & C_bovis_copy_sample == 0 & Str_uberis_copy_sample == 0 ~ "F*",  
    TRUE ~ "Unclassifiable"  
  ))
milk_data$Corrected_Profile <- as.factor(milk_data$Corrected_profiles)

ggplot(milk_data, aes(x = Corrected_profiles, fill = Profiles)) +
  geom_bar() 

group_summary_corrected <- milk_data %>%
  group_by(Corrected_profiles) %>%
  summarise(Nb_sample = n())

milk_data <- milk_data %>%
  left_join(group_summary_corrected, by = "Corrected_profiles") %>%
  mutate(Label_corrected_profiles = paste0(Corrected_profiles, " (n=", Nb_sample, ")"))

milk_data_long_corrected <- milk_data %>%
  select(ends_with("copy_sample"), -Beta_lacta_copy_sample, Corrected_profiles, Date, Label_corrected_profiles) %>%
  pivot_longer(cols = ends_with("copy_sample"), 
               names_to = "Pathogen", 
               values_to = "Copy")

cols_corrected <- c(
  "C*" = "#66C2A5",
  "D*" = "#FC8D62",
  "A*" = "#8DA0CB",
  "F*" = "#E78AC3",
  "E*" = "#A6D854",
  "B*" = "#FFD92F",
  "Unclassifiable" = "#BBBBBB"
)

ordered_cols_corrected <- cols_corrected[order(names(cols_corrected))]
strip <- strip_themed(background_x = elem_list_rect(fill = ordered_cols_corrected))

graph_patho <- ggplot(milk_data_long_corrected, aes(x = Pathogen, y = Copy + 1)) +
  geom_quasirandom(size=0.1) +
  theme_bw()+
  scale_x_discrete(labels = Pathogen_name) +
  scale_y_log10() +
  ylab("Pathogen quantity") + 
  theme(axis.title.y = element_text(size = 10))+
  facet_wrap2(~Label_corrected_profiles, ncol = 2, dir = "h", strip = strip) +
  theme(axis.text.x = element_text(angle = 90, 
                                   # hjust = 1, 
                                   vjust = 0.5),
        legend.position = "none",
        strip.text = element_text(size = 10),  
        axis.text=element_text(size=10)) +
  scale_fill_manual(values = color_palette)+
  theme(legend.position = "none")
# ggsave("supp_mat7.pdf", plot = graph_patho , width = 170, height = 150, units = "mm" ) 
# rm(graph_patho, group_summary_corrected)

# Create the confusion matrix
confusion_matrix <- table(Profiles = milk_data$Profiles, Corrected_profiles = milk_data$Corrected_profiles)

misclassified <- sum(milk_data$Profiles != substr(milk_data$Corrected_profiles, 1, 1))
misclassification_rate <- misclassified / nrow(milk_data) *100 
misclassification_rate

Unclassifiable_profil <- milk_data%>%
  select(Lab_ID, Cow, Corrected_profiles, ends_with("copy_sample"))%>%
  filter(Corrected_profiles == "Unclassifiable")
rm(misclassification_rate, misclassified, Unclassifiable_profil, confusion_matrix)


#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   d. Clustering transition for Profiles VS for Corrected_profiles
#--------------------------------------------------------------

### Profiles transitions
data_transition_Milk_profiles <- milk_data %>%
  select(Date, Cow, Profiles, Lab_ID) %>%
  mutate(
    Date = as.Date(Date, format = "%d/%m/%Y"),
    Cow = as.factor(Cow),
    Lab_ID = as.factor(Lab_ID),
    Profiles = as.factor(Profiles)) %>%
  arrange(Date)

sequences <- list()
List_cow <- unique(milk_data$Cow)

for (c in List_cow) {
  data_c <- data_transition_Milk_profiles[data_transition_Milk_profiles$Cow == c, ]
  data_c <- data_c[order(data_c[, 1]), ]
  sequence_cow <- data_c$Profiles
  sequences[[as.character(c)]] <- sequence_cow
}

Q_Milk<-markovchainFit(data=sequences, method = "mle", byrow=T)
Matrices_Milk<-Q_Milk$estimate@transitionMatrix
steadyStates(Q_Milk$estimate)

Profiles_transition <- melt(Matrices_Milk, varnames = c("From", "To"), value.name = "Probability_profiles")
# write.csv2(Profiles_transition,"Profiles_transition.csv") # -> Script Transition_graph_V2.py
# use the python script: Transition_graph.py to obtain the corresponding figure
rm(data_c, sequence_cow, sequences, c, Q_Milk, Matrices_Milk)

#############
### Corrected profiles transitions
data_transition_Milk_corrected_profiles <- milk_data %>%
  select(Date, Cow, Corrected_profiles, Lab_ID) %>%
  mutate(
    Date = as.Date(Date, format = "%d/%m/%Y"),
    Cow = as.factor(Cow),
    Lab_ID = as.factor(Lab_ID),
    Corrected_profiles = na_if(Corrected_profiles, "Unclassifiable")) %>%
  arrange(Date)

List_cow <- unique(data_transition_Milk_corrected_profiles$Cow)
sequences_corrected <- list()

for (c in List_cow) {
  data_corrected_c <- data_transition_Milk_corrected_profiles[data_transition_Milk_corrected_profiles$Cow == c, ]
  data_corrected_c <- data_corrected_c[order(data_corrected_c[, 1]), ]
  sequence_cow <- data_corrected_c$Corrected_profiles
  sequences_corrected[[as.character(c)]] <- sequence_cow
}

Q_Milk_corrected<-markovchainFit(data=sequences_corrected, method = "mle", byrow=T)
Matrices_Milk_corrected<-Q_Milk_corrected$estimate@transitionMatrix
steadyStates(Q_Milk_corrected$estimate)

Corrected_profiles_transition <- melt(Matrices_Milk_corrected, varnames = c("From", "To"), value.name = "Probability_state")
# write.csv2(Corrected_profiles_transition,"Corrected_profiles_transition.csv") # -> Script Transition_graph_V2.py
rm(data_corrected_c, sequence_cow, sequences_corrected, c, Q_Milk_corrected,Matrices_Milk_corrected)


########
## Comparison of transitions
Transition <- Corrected_profiles_transition %>%
  mutate(From = str_remove_all(From, "\\*"),
         To = str_remove_all(To, "\\*"))
Transition_all <- merge(Transition, Profiles_transition, by=c("From","To"))

plot_comparison_transition <- ggplot(Transition_all,aes(x=Probability_profiles, y=Probability_state))+
  geom_point()+
  # geom_smooth(method = "lm", se = FALSE, color = "blue") +
  theme_minimal()+
  xlab("Probability of transtion between Profiles")+
  ylab("Probability of transtion between corrected Profiles")
# ggsave("supp_mat8.pdf", plot = plot_comparison_transition , width = 170, height = 120, units = "mm" ) 
rm(Transition, Transition_all, plot_comparison_transition)


########
### Profile over time

dates_observed <- milk_data %>%
  select(Farm, Date)%>%
  filter(!is.na(Farm)) %>%
  distinct(Farm, Date)

cows_per_farm <- milk_data %>%
  select(Cow, Farm)%>%
  filter(!is.na(Farm)) %>%
  distinct(Farm, Cow)

grid <- dates_observed %>%
  left_join(cows_per_farm, by = "Farm", relationship = "many-to-many") %>%
  left_join(milk_data, by = c("Farm", "Cow", "Date"))

data_ordered <- grid %>%
  group_by(Cow) %>%
  arrange(Date, .by_group = TRUE) %>%
  mutate(Sample_ID = row_number()) %>%
  ungroup()

data_ordered$Antibiotic_administration <- factor(
  data_ordered$Antibiotic_administration,
  levels = c("Intramammary", "Other", "Intramammary_Other"),
  labels = c("Intramammary", "Systemic", "Intramammary + Systemic"))

plot_profile_over_time <- ggplot(data_ordered, aes(x = Sample_ID*3.5-1, y = Cow, fill = Profiles)) +
  geom_tile(width = 3.5, height = 0.85) + 
  geom_point(data = subset(data_ordered, !is.na(Antibiotic_administration)),
             aes(shape = Antibiotic_administration, x=(Sample_ID -0.5)*3.5-1, color=Antibiotic_family),
             size = 1.5, position = position_nudge(y = 0)) +
  scale_shape_manual(values = c("Intramammary" = 17,"Systemic" = 16,"Intramammary + Systemic" = 15), 
                      guide = guide_legend(ncol = 1, title = 'Antibiotic administration', override.aes = list(size = 2)))+
  scale_color_manual(values = c("Aminopenicillin" = 'green', "Tetracycline" = 'blue', 
                                "Sulfonamide" = 'grey95', "Macrolide" = 'cyan2', 
                                "Cephalosporin" = 'red', "Penicillin+Aminoglycoside" = 'gold2',
                                "Sulfonamide+Cephalosporin" = 'black'),
                     guide = guide_legend(ncol = 1, title = 'Antibiotic family')) +
  scale_fill_manual(values = color_palette, na.value = "grey80",
                    guide = guide_legend(fill='none')) +
  labs(x = "Time (number of days since first sample)", y = "Cow", fill = "Profiles") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 7), 
        axis.text.x = element_text(size = 5), 
        axis.ticks.x = element_line(linewidth = 0.1, colour = "grey"),     
        panel.grid = element_blank())+
  guides(fill = "none")+
  scale_x_continuous(
    breaks = seq(0, max(data_ordered$Sample_ID * 3.5, na.rm = TRUE), by = 10) )

legend_plot <- ggplot(data_ordered, aes(x = Sample_ID*3.5, y = Cow, fill = Profiles)) +
  geom_tile(width = 1, height = 0.85) + 
  scale_fill_manual(values = color_palette, na.value = "grey80") 
legend <- get_legend(legend_plot)  

profiles_over_time <- ggarrange(plot_profile_over_time, legend, ncol=2,widths = c(0.92,0.08))


# ggsave("Figure_supp_mat_6.pdf", plot = profiles_over_time , width = 170, height = 100, units = "mm" ) 



#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   d. Clustering transition for Profiles based on presence/absence of NAS 
#--------------------------------------------------------------

milk_data <- milk_data %>%
  mutate(Profile_NAS = case_when(
    Profiles == "C" & NAS_copy_sample > 0 ~ "C_NAS",
    Profiles == "C" & NAS_copy_sample == 0 ~ "C_no_NAS",
    Profiles == "D" & NAS_copy_sample > 0 ~ "D_NAS",
    Profiles == "D" & NAS_copy_sample == 0 ~ "D_no_NAS",
    Profiles == "E" & NAS_copy_sample > 0 ~ "E_NAS",
    Profiles == "E" & NAS_copy_sample == 0 ~ "E_no_NAS",
    Profiles == "F" & NAS_copy_sample > 0 ~ "F_NAS",
    Profiles == "F" & NAS_copy_sample == 0 ~ "F_no_NAS",
    TRUE ~ as.character(Profiles)
    ))

data_transition_Milk_profiles_NAS <- milk_data %>%
  select(Date, Cow, Profile_NAS, Lab_ID) %>%
  mutate(
    Date = as.Date(Date, format = "%d/%m/%Y"),
    Cow = as.factor(Cow),
    Lab_ID = as.factor(Lab_ID),
    Profile_NAS = na_if(Profile_NAS, "Unclassifiable")) %>%
  arrange(Date)

List_cow <- unique(milk_data$Cow)
sequences_corrected <- list()

for (c in List_cow) {
  data_corrected_c <- data_transition_Milk_profiles_NAS[data_transition_Milk_profiles_NAS$Cow == c, ]
  data_corrected_c <- data_corrected_c[order(data_corrected_c[, 1]), ]
  sequence_cow <- data_corrected_c$Profile_NAS
  sequences_corrected[[as.character(c)]] <- sequence_cow
}

Q_Milk_corrected<-markovchainFit(data=sequences_corrected, method = "mle", byrow=T)
Matrices_Milk_corrected<-Q_Milk_corrected$estimate@transitionMatrix
steadyStates(Q_Milk_corrected$estimate)

Corrected_profiles_transition_NAS <- melt(Matrices_Milk_corrected, varnames = c("From", "To"), value.name = "Probability_state")
# write.csv2(Corrected_profiles_transition_NAS,"Profiles_transition_NAS.csv") # -> Script Transition_graph_NAS.py




#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   e. Effect of treatment on transitions
#--------------------------------------------------------------

data_transition_antibiotic <-  data.frame()
                      
# Processing each cow
for (c in List_cow) {
  # Milk_filter the data for each cow
  data_c <- data_transition_Milk_profiles[data_transition_Milk_profiles$Cow == c, ]
  data_c <- data_c[order(data_c$Date), ]
  # Create columns Profiles and Profiles_before
  data_c$Profiles_after1 <- c(as.character(data_c$Profiles[-1]), NA)
  data_c$Profiles_after2 <- c(as.character(data_c$Profiles[-c(1,2)]), NA, NA)
  data_c$Profiles_before1 <- c(NA, head(as.character(data_c$Profiles), -1))  
  data_c$Profiles_before2 <- c(NA, NA, head(as.character(data_c$Profiles), -2))
  # ATB data with NA
  milk_data_c <- milk_data%>%
    filter(Cow == c)%>%
    select(Lab_ID, Antibiotic)
  milk_data_c_all <- merge(milk_data_c, data_c, all = TRUE)
  milk_data_c_all <- milk_data_c_all[order(milk_data_c_all$Date), ]
  ### Assign ATB 
  data_c <- data_c %>%
    left_join(milk_data_c_all %>% select(Date, Antibiotic),
              by = "Date") %>%
    rename(ATB = Antibiotic)
  
  data_c$ATB_before1 <- c(NA, head(milk_data_c_all$Antibiotic, -1))  
  data_c$ATB_before2 <- c(NA, NA, head(milk_data_c_all$Antibiotic, -2))
  # Add these data to the final table
  data_transition_antibiotic <- rbind(data_transition_antibiotic, data_c)
}

data_transition_antibiotic_summary <- data_transition_antibiotic %>%
  filter(ATB == "Yes") %>% 
  mutate(From = as.character(ifelse(is.na(Profiles_before1), Profiles_before2, Profiles_before1)),
         To = as.character(ifelse(is.na(as.character(Profiles)), as.character(Profiles_after1), as.character(Profiles)))) %>%
  select(From, To) %>%
  count(From, To, name = "Number") %>%  
  arrange(desc(Number)) 

# write.csv2(data_transition_antibiotic_summary,"Antibiotic_transition.csv")
# use the python script: Transition_antibiotic_V2.py to obtain the corresponding figure
rm(c, data_c, milk_data_c, milk_data_c_all, data_transition_antibiotic_summary)


#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   f. Effect of somatic cells on transitions
#--------------------------------------------------------------

data_transition_cell <- data.frame()

# Process each cow
for (c in List_cow) {
  data_c <- data_transition_Milk_profiles[data_transition_Milk_profiles$Cow == c, ]
  data_c <- data_c[order(data_c$Date), ]  
  # Create the 'Profiles_before1' column (shifted by one row to get previous profile)
  data_c$Profiles_before1 <- c(NA, head(as.character(data_c$Profiles), -1))
  # Filter the somatic cells data for the cow 'c'
  data_cell_c <- milk_data[milk_data$Cow == c, ]
  data_cell_c <- data_cell_c[order(data_cell_c$Date), ]  
  # Create lagged cell columns (previous somatic cells)
  data_c$cell_before2 <- as.numeric(c(NA, NA, head(data_cell_c$Somatic_cells_interpolation, -2)))  # Lagged by 2
  data_c$cell_before1 <- as.numeric(c(NA, head(data_cell_c$Somatic_cells_interpolation, -1)))  # Lagged by 1
  data_c$cell <- as.numeric(data_cell_c$Somatic_cells_interpolation)  # Current somatic cells
  # Calculate the time differences (Delta_day) between observations
  data_c$Delta_day_0_1 <- c(NA, diff(data_c$Date))  # Time difference between current and previous observation
  data_c$Delta_day_1_2 <- c(NA, NA, head(diff(data_c$Date), -1))  # Time difference between previous and second previous observation
  
  # Add the data for the cow to the final table
  data_transition_cell <- rbind(data_transition_cell, data_c)
}
rm(c, data_c, data_cell_c)

# Calculate the change in somatic cells over time for analysis
data_transition_cell$Delta_cell_before1_before2 <- (data_transition_cell$cell_before1 - data_transition_cell$cell_before2) / data_transition_cell$Delta_day_1_2
data_transition_cell$Delta_cell_day1_before1 <- (data_transition_cell$cell - data_transition_cell$cell_before1) / data_transition_cell$Delta_day_0_1

# Create new variables indicating whether the cow has transitioned between profiles (from before to current)
data_cell <- data_transition_cell %>%
  mutate(
    Transition_to_A = as.factor(ifelse(Profiles == "A" & Profiles_before1 != "A", "Yes", "No")),
    Transition_to_B = as.factor(ifelse(Profiles == "B" & Profiles_before1 != "B", "Yes", "No")),
    Transition_to_C = as.factor(ifelse(Profiles == "C" & Profiles_before1 != "C", "Yes", "No")),
    Transition_to_D = as.factor(ifelse(Profiles == "D" & Profiles_before1 != "D", "Yes", "No")),
    Transition_to_E = as.factor(ifelse(Profiles == "E" & Profiles_before1 != "E", "Yes", "No")),
    Transition_to_F = as.factor(ifelse(Profiles == "F" & Profiles_before1 != "F", "Yes", "No"))
  ) %>%
  filter(Profiles_before1 != "NA") %>%
  filter(Profiles != Profiles_before1)  # Ensure there is a profile change

# Create a summary table for t-test results
results <- list()

# Perform the t-test for each Transition_to (A to F)
Transitions_to <- c("Transition_to_A", "Transition_to_B", "Transition_to_C", "Transition_to_D", "Transition_to_E", "Transition_to_F")

for (Transition_to in Transitions_to) {
  # Run the t-test for the 'Delta_cell_before1_before2' variable and Transition_to
  t_test_1 <- t.test(data_cell$Delta_cell_before1_before2 ~ data_cell[[Transition_to]])
  
  # Run the t-test for the 'Delta_cell_day1_before1' variable and Transition_to
  t_test_2 <- t.test(data_cell$Delta_cell_day1_before1 ~ data_cell[[Transition_to]])
  
  # Save the results in the list
  results[[Transition_to]] <- data.frame(
    Transition_to = Transition_to,
    t_statistic_cell_before1_before2 = t_test_1$statistic,
    df_cell_before1_before2 = t_test_1$parameter,
    p_value_cell_before1_before2 = t_test_1$p.value,
    t_statistic_cell_0_before1 = t_test_2$statistic,
    df_cell_0_before1 = t_test_2$parameter,
    p_value_cell_0_before1 = t_test_2$p.value
  )
}

# Combine all the results into a final table
results_table <- bind_rows(results)

# Display the results (optional)
print(results_table)
rm(Transition_to, Transitions_to, t_test_1, t_test_2, data_cell,results_table)


#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   g. Clustering robustness
#--------------------------------------------------------------

# Perform hierarchical clustering
Group_full <- cutree(ward_cluster, k = 6) 
data_full <- data.frame(Group_full, Lab_ID = milk_data$Lab_ID)

# Assign group names
data_full <- data_full %>%
  mutate(group = case_when(
    Group_full == 1 ~ "C",
    Group_full == 2 ~ "A",
    Group_full == 3 ~ "B",
    Group_full == 4 ~ "D",
    Group_full == 5 ~ "E",
    Group_full == 6 ~ "F"))

############### Removing 10% of the data 
# Total number of samples (rows) in milk_data
n <- nrow(milk_data)
# Randomly select 10% of the row indices
set.seed(123)  
indices_to_remove <- sample(1:n, size = floor(0.2 * n))

# Remove the selected rows from the data frame
milk_data_filtered_removed <- milk_data[-indices_to_remove, ]
milk_data_filtered_removed <- milk_data_filtered_removed %>%
  select("Lab_ID", "Cow", "Date", ends_with("copy_sample"), - Beta_lacta_copy_sample)

# Create a matrix for clustering and calculate distances
milk_data_matrix_removed <- data.matrix(log(milk_data_filtered_removed[, 4:16] + 1))
dres_removed <- dist(milk_data_matrix_removed, method = "euclidean") 
myclust_removed <- hclust(dres_removed, method = "ward.D2")

# Perform clustering and assign groups
Group_removed <- cutree(myclust_removed, k = 6) 
without <- data.frame(Group_removed, milk_data_filtered_removed$Lab_ID)

## Clustering - Description of the groups 
milk_data_without_removed <- milk_data_filtered_removed
milk_data_without_removed$Group <- Group_removed

# Pivot data from wide to long format
milk_data_without_removed_long <- milk_data_without_removed %>%
  pivot_longer(cols = c("C_bovis_copy_sample", "E_coli_copy_sample", "Entero_sp_copy_sample", 
                        "Klebsie_sp_copy_sample", "Myco_sp_copy_sample", "Proto_sp_copy_sample", 
                        "Staph_aureus_copy_sample", "NAS_copy_sample", "Str_agala_copy_sample", 
                        "Str_dysgala_copy_sample", "Str_uberis_copy_sample", "T_pyo_P_indo_copy_sample", 
                        "Yeasts_copy_sample"),
               names_to = "Pathogen", 
               values_to = "copy")

# Add the modified column
milk_data_without_removed_long <- milk_data_without_removed_long %>%
  mutate(copy_modified = copy + 1)

# Now you can proceed with the rest of the code, for example, plotting
ggplot(milk_data_without_removed_long, aes(x = Pathogen, y = copy_modified, fill = Pathogen)) +
  geom_violin(scale = "width") +
  ylab("Groups") +
  scale_y_log10() +
  facet_wrap(~Group, nrow = 3)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))


# Create final data frame for comparison
Group_removed <- cutree(myclust_removed, k = 6) 
data_without_removed <- data.frame(Group_removed, Lab_ID = milk_data_filtered_removed$Lab_ID)

#  Assign names to the groups according to the characteristics of the groups so that they correspond to the groups with the data set
data_without_removed <- data_without_removed %>%
  mutate(without = case_when(
    Group_removed == 1 ~ "D",
    Group_removed == 2 ~ "A",
    Group_removed == 3 ~ "B",
    Group_removed == 4 ~ "C",
    Group_removed == 5 ~ "F",
    Group_removed == 6 ~ "E"))

# Merge the original and filtered data
data_merged <- merge(data_full, data_without_removed, by = "Lab_ID")
data_merged$diff <- ifelse(data_merged$group == data_merged$without10, "No", "Yes")

# Summary of the differences
summary(as.factor(data_merged$diff))

# Calculate the percentage of "No" differences
percentage_diff <- sum(data_merged$diff == "No") * 100 / nrow(data_merged)
percentage_diff


#--------------------------------------------------------------
# 3. Ward clustering of milk data
#   h. Duration of infection by pathogens in milk samples 
#--------------------------------------------------------------
data_infection <- milk_data %>%
  select(Lab_ID, Cow, Date, Matrices, ends_with("copy_sample"), -Beta_lacta_copy_sample)

pathogen_columns <- c("C_bovis_copy_sample", "E_coli_copy_sample", "Entero_sp_copy_sample", "Klebsie_sp_copy_sample", 
                      "Myco_bovis_copy_sample", "Myco_sp_copy_sample", "S_marces_copy_sample", "Proto_sp_copy_sample", 
                      "Staph_aureus_copy_sample", "NAS_copy_sample", "Str_agala_copy_sample", "Str_dysgala_copy_sample", 
                      "Str_uberis_copy_sample", "T_pyo_P_indo_copy_sample", "Yeasts_copy_sample")

data_infection_long <- data_infection %>%
  pivot_longer(cols = all_of(pathogen_columns), names_to = "pathogen", values_to = "Copy_sample") %>%
  mutate(pathogen = gsub("_copy_sample", "", pathogen), 
         Date = as.Date(Date, format = "%d/%m/%Y")) %>%
  arrange(Cow, pathogen, Date) 

data_infection_long$Pathogene_presence_absence <- ifelse(data_infection_long$Copy_sample>0, 1, 0)
data_infection_long$Pathogene_presence_absence[is.na(data_infection_long$Pathogene_presence_absence)] <- 0

results <- data.frame(Cow = character(), pathogen = character(), infection_duration = integer(), stringsAsFactors = FALSE)

# Boucle sur chaque cow
for (c in unique(data_infection_long$Cow)) {
  data_c <- data_infection_long %>% filter(Cow == c)
  
  for (patho in unique(data_c$pathogen)) {
    data_patho <- data_c %>% filter(pathogen == patho) %>%
      arrange(Date)  
    
    data_patho_clean <- data_patho %>%
      mutate(ppa_raw = ifelse(is.na(Copy_sample), NA, ifelse(Copy_sample > 0, 1, 0))) %>%
      
      mutate(ppa_before = lag(ppa_raw)) %>%
      fill(ppa_before, .direction = "down") %>%
      
      mutate(ppa_after = lead(ppa_raw)) %>%
      fill(ppa_after, .direction = "up") %>%
      
      mutate(Pathogene_fixed = case_when(
        is.na(Copy_sample) & ppa_before == 1 & ppa_after == 1 ~ 1,
        is.na(Copy_sample) ~ 0,
        TRUE ~ Pathogene_presence_absence)) %>%
      mutate(infection_group = cumsum(
        coalesce(Pathogene_fixed != lag(Pathogene_fixed, default = first(Pathogene_fixed)), FALSE))) %>%
      select(-ppa_before, -ppa_after, -Pathogene_presence_absence)
    
    if (nrow(data_patho_clean) > 0) {
      infection_summary <- data_patho_clean %>%
        group_by(infection_group) %>%
        filter(Pathogene_fixed ==1)%>%
        summarise(
          Cow = first(Cow),
          pathogen = first(pathogen),
          infection_duration = n(),
          .groups = "drop"
        ) %>%
        select(Cow, pathogen, infection_duration)
      
      # Ajouter les résultats au dataframe principal
      results <- bind_rows(results, infection_summary)
    }
  }
}

results%>%
  select(!Cow)%>%
  tbl_summary(by=pathogen,statistic = list(
    all_continuous() ~ "{min}, {mean}, {median}, {max} ({sd})"))

results$infection_duration <- as.factor(results$infection_duration)

counts <- results %>%
  group_by(pathogen, infection_duration) %>%
  summarise(count = n(), .groups = 'drop')

tableau_counts <- counts %>%
  pivot_wider(names_from = infection_duration, values_from = count, values_fill = 0)

all_durations <- as.character(sort(unique(results$infection_duration)))
results$infection_duration <- factor(results$infection_duration, levels = all_durations)

counts <- results %>%
  group_by(pathogen, infection_duration) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(pathogen) %>%
  mutate(
    total_cases = sum(count),
    proportion = count *100 / total_cases
  ) %>%
  ungroup()

pathogen_max_counts <- counts %>%
  group_by(pathogen) %>%
  summarise(max_infection = max(as.numeric(infection_duration), na.rm = TRUE)) %>%
  filter(max_infection <= 1) %>%
  pull(pathogen)

median_order <- counts %>%
  mutate(infection_duration = as.numeric(as.character(infection_duration))) %>%
  group_by(pathogen) %>%
  summarise(median_duration = median(infection_duration, na.rm = TRUE)) %>%
  arrange(desc(median_duration))

counts_complete <- counts %>%
  mutate(infection_duration = as.numeric(as.character(infection_duration))) %>%
  filter(!pathogen %in% pathogen_max_counts) %>%   # <-- aussi ici
  complete(pathogen, infection_duration, fill = list(count = NA_real_)) %>%
  left_join(median_order, by = "pathogen") %>%
  mutate(pathogen = factor(pathogen, levels = median_order$pathogen))

Pathogen_name <- c('C_bovis' = expression(italic("C. bovis")),
                   'E_coli' = expression(italic("E. coli")),
                   'Entero_sp' = expression(italic("Enterococcus")~" spp."),
                   'Klebsie_sp' = expression(italic("Klebsiella")~" spp."),
                   'Myco_bovis' = expression(italic("M. bovis")),
                   'Myco_sp' = expression(italic("Mycoplasma")~" spp."),
                   'Proto_sp' = expression(italic("Prototheca")~" spp."),
                   'S_marces' = expression(italic("S. marcescens")),
                   'Staph_aureus' = expression(italic("S. aureus")),
                   'NAS' = expression("NAS"),
                   'Str_agala' = expression(italic("S. agalactiae")),
                   'Str_dysgala' = expression(italic("S. dysgalactiae")),
                   'Str_uberis' = expression(italic("S. uberis")),
                   'T_pyo_P_indo' = expression(italic("T. pyogenes/P. indolicus")),
                   'Yeasts' = "Yeasts")

duration_infection <- ggplot(counts_complete, aes(x = (infection_duration * 3.5), y = pathogen, fill = as.numeric(count))) +
  geom_tile(color = "white", width = 1.3) +
  scale_fill_viridis_c(trans = "log",
                       option = "D",
                       breaks = c(1, 2, 5, 10, 30, 80),
                       na.value = "transparent", direction = -1) +
  labs(x = "Duration of infection (days)", y = "Pathogen", fill = "Number of occurrences") +
  theme_minimal() +
  theme(axis.ticks = element_line(linewidth = 0.2, colour = "grey"), 
        panel.grid.major.y = element_line(linewidth = 0.2),
        panel.grid.major.x = element_line(linewidth = 0.2),
        panel.grid.minor.x = element_line(linewidth = 0.2),
        axis.text=element_text(size=6),
        axis.title=element_text(size=8),
        legend.text = element_text(size = 6),
        legend.title = element_text(size = 8),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.3)) +
  scale_y_discrete(labels = Pathogen_name)+
  scale_x_continuous(
    breaks = seq(00, (max(counts_complete$infection_duration * 3.5) + 10), by = 10),
    limits = c(0.5, (max(counts_complete$infection_duration * 3.5) + 5)))

# ggsave("Figure_9.pdf", plot = duration_infection , width = 170, height = 80, units = "mm" ) 

