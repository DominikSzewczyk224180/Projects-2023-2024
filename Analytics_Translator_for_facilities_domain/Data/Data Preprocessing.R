library(dplyr)
library(tidyverse)
library(ggplot2)
library(forcats)
library(plotly)
library(dplyr)
library(broom)
library(MVN)
library(agricolae)
library(ggcorrplot) 
library(modelr)   
library(broom)
library(viridis)
library(gghighlight)
library(hrbrthemes)
library(viridis)
library(gridExtra)
  
# Reading both numerical and text df
df_main_text <-read.csv("C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/survey_text_27_10.csv",header = TRUE,sep = ",")
df_main_num <- read.csv("C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/survey_num_27_10.csv",header = TRUE,sep = ",")
df_main_text[df_main_text == ""] <- NA
df_main_num[df_main_num == ""] <- NA

# filtering df for "Facility"
df_text <- df_main_text %>% 
  slice(-1:-2) %>%
  filter(demo_domain == 'Facility')
df_num <- df_main_num %>%
  slice(-1:-2) %>%
  filter(demo_domain == '8')


# cleaning and subsetting datasets
df_ai_general_num <- df_num %>%
  select(c(12:15,17:36)) %>% 
  mutate_at(vars(-c(1:6)),as.numeric) %>%
  na.omit()
df_pred_m_num <- df_num %>%
  select(c(12:15,17,18,268:270,272:273)) %>%
  mutate_at(vars(-c(1:6,'f_predmain_method')),as.numeric) %>%
  na.omit()
df_campus_num <- df_num %>%
  select(c(12:15,17,18,255:257,259,261:267)) %>%
  mutate_at(vars(-c(1:6)),as.numeric) %>%
  na.omit()
df_ai_impact_num <- df_num %>%
  select(c(12:15,17,18, 249:253)) %>%
  mutate_at(vars(-c(1:6, 'f_aifm_key_roles')),as.numeric) %>%
  na.omit()
df_space_opt_num <- df_num %>%
  select(c(12:15,17,18, 274:277)) %>%
  mutate_at(vars(-c(1:6)),as.numeric) %>%
  na.omit()
df_sec_num <- df_num %>%
  select(c(12:15,17,18, 278:286)) %>%
  mutate_at(vars(-c(1:6,'f_sec_preference')),as.numeric) %>%
  na.omit()


df_ai_general_text <- df_text %>%
  select(c(12:15,17:36)) %>% 
  na.omit()
df_pred_m_text <- df_text %>%
  select(c(12:18,268:270,272:273)) %>%
  na.omit()
df_campus_text <- df_text %>%
  select(c(12:15,17,18,255:257,259,261:267)) %>%
  na.omit()
df_ai_impact_text <- df_text %>%
  select(c(12:15,17,18, 249:253)) %>%
  na.omit()
df_space_opt_text <- df_text %>%
  select(c(12:15,17,18, 274:277)) %>%
  na.omit()
df_sec_text <- df_text %>%
  select(c(12:15,17,18, 278:286)) %>%
  na.omit()

  
#exporting preprocessed datasets

#numerical data
write.csv(df_ai_general_num, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_ai_general_num.csv",row.names=FALSE)
write.csv(df_pred_m_num, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_pred_m_num.csv",row.names=FALSE)
write.csv(df_campus_num, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_campus_num.csv",row.names=FALSE)
write.csv(df_ai_impact_num, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_ai_impact_num.csv",row.names=FALSE)
write.csv(df_space_opt_num, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_space_opt_num.csv",row.names=FALSE)
write.csv(df_sec_num, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_sec_num.csv",row.names=FALSE)


#textual data
write.csv(df_ai_general_text, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_ai_general_text.csv",row.names=FALSE)
write.csv(df_pred_m_text, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_pred_m_text.csv",row.names=FALSE)
write.csv(df_campus_text, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_campus_text.csv",row.names=FALSE)
write.csv(df_ai_impact_text, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_ai_impact_text.csv",row.names=FALSE)
write.csv(df_space_opt_text, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_space_opt_text.csv",row.names=FALSE)
write.csv(df_sec_text, file = "C:/Users/NITRO/Desktop/2023-24a-fai2-adsai-SimonaDimitrova222667/scripts/df_sec_text.csv",row.names=FALSE)

  
  
  
  
  
  
  
  
  
  
