library('tidyverse')
library('ggplot2')
library('GGally')
library('ggsci')
library('Hmisc')
library('ggcorrplot')
library('praznik')
#load data
pelagic_df = read.csv('D:/program/aquatic/data/pelagic_training_set.csv')

view(pelagic_df)
head(pelagic_df)
pelagic_df_num = pelagic_df |> select_if(is.numeric)

#overview of pelagic fish plot
ggpairs(pelagic_df_num,
        upper = list(continuous = wrap("cor", size = 4, colour = "darkblue")),
        lower = list(continuous = wrap("points", alpha = 0.3, size = 0.8)),
        diag = list(continuous = wrap("densityDiag", fill = "steelblue", alpha = 0.6)),
        title = "Pelagic Fish Dataset - Correlation Matrix",
        axisLabels = "show") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    strip.background = element_rect(fill = "lightgray", color = "gray40"),
    strip.text = element_text(face = "bold", size = 10)
  )
##Note
#--distribution--
#observed depth: right-skewed (mostly shallow water)
#body length: unimodal, mostly right-skewed(light)
#body weight: heavily right skewed
#water temperature and salinity: bimodal, two extreme side
#pH: above 7, left-skewed, alkaline
#estimated age: right-skewed (young)

#--correlation--
#body weight and depth: little positive linear relationship(r = 0.483)
#body weight and body length: high linear relationship(r=0.821)
#water temperature and depth: negative linear relationship(r=-0.676)
#age and weight: some positive linear relationship(r=0.583)

#spearman r correlation

pearson = rcorr(as.matrix(pelagic_df_num), type='spearman')
r_matrix = pearson$r
p_matrix = pearson$p
#corr plot
ggcorrplot(r_matrix, 
           method = "square",
           type = "lower",
           ggtheme = ggplot2::theme_minimal,
           title = "Pelagic Species Spearman Correlation Matrix",
           lab = TRUE,
           lab_size = 3,
           p.mat = p_matrix,
           sig.level = 0.05,
           insig = "blank",
           colors = c("#f7fbfd", "#4d9eb5", "#2e7d8a", "#1a5c6e", "#0d3b4f"),  # Lighter ocean palette
           outline.color = "white",
           tl.cex = 10,
           tl.col = "#1a3a4a") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", color = "#0d3b4f"),
    panel.background = element_rect(fill = "#e6f2f5", color = NA),  # Light ocean mist
    plot.background = element_rect(fill = "#e6f2f5", color = NA),
    # Improve number visibility
    text = element_text(color = "#0a2a3a"),
    axis.text = element_text(color = "#0a2a3a", size = 10),
    legend.title = element_text(color = "#0a2a3a", size = 10),
    legend.text = element_text(color = "#0a2a3a", size = 9)
  )
#note: spearman r shows roughly same trend as pearson

#mutual information
MI = miMatrix(pelagic_df_num)
ggcorrplot(MI,
           method = "square",
           type = "lower",
           ggtheme = ggplot2::theme_minimal,
           title = "Mutual Information Matrix",
           lab = TRUE,
           lab_size = 3,
           colors = c("#f7fbfd", "#4d9eb5", "#2e7d8a", "#1a5c6e", "#0d3b4f"),
           outline.color = "white",
           tl.cex = 10,
           tl.col = "#1a3a4a") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", color = "#0d3b4f"),
    panel.background = element_rect(fill = "#e6f2f5", color = NA),
    plot.background = element_rect(fill = "#e6f2f5", color = NA)
  )



#normalized mutual information matrix
NMI = nmiMatrix(pelagic_df_num)   
ggcorrplot(NMI,
           method = "square",
           type = "lower",
           ggtheme = ggplot2::theme_minimal,
           title = "Normalized Mutual Information Matrix",
           lab = TRUE,
           lab_size = 3,
           colors = c("#f7fbfd", "#4d9eb5", "#2e7d8a", "#1a5c6e", "#0d3b4f"),
           outline.color = "white",
           tl.cex = 10,
           tl.col = "#1a3a4a") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", color = "#0d3b4f"),
    panel.background = element_rect(fill = "#e6f2f5", color = NA),
    plot.background = element_rect(fill = "#e6f2f5", color = NA)
  )

