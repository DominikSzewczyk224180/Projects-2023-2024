library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
data <- read_xlsx("C:/Users/domin/Desktop/df_text_2023-10-22.xlsx") 

View(data_facility) 
data_facility <- data 
data_facility$StartDate <- NULL 
data_facility$EndDate <- NULL 
data_facility$Status <- NULL 
data_facility$Progress <- NULL 
data_facility$Finished <- NULL 
data_facility$RecordedDate <- NULL 
data_facility$ResponseId <- NULL 
data_facility$DistributionChannel <- NULL 
data_facility$UserLanguage <- NULL 

data_facility <- data_facility[!(data_facility$demo_domain %in% c("Built Environment", "Tourism","Other","Media","Logistics","Leisure & Events", "Hotel","Games","Click to write Choice 10","") | is.na(data_facility$demo_domain)), ] 
data_facility <- data_facility[-2, ]

row_to_check <- 3

# Remove columns with NA values in the specified row
na_cols <- which(is.na(data_facility[row_to_check, ]))

data_facility <- data_facility[, -na_cols, drop = FALSE]

# PLOT

my_colors <- c(
  "Strongly agree" = "#3498db",
  "Somewhat agree" = "#2ecc71",
  "Neither agree nor disagree" = "#f1c40f",
  "Somewhat disagree" = "#e74c3c",
  "Strongly disagree" = "#9b59b6"
)
my_colors_2 <- c(
  "Extremely positive" = "#3498db",
  "Somewhat positive" = "#2ecc71",
  "Neither positive nor negative" = "#f1c40f",
  "Somewhat negative" = "#e74c3c",
  "Extremely negative" = "#9b59b6"
)
my_colors_3 <- c(
  "Definitely yes" = "#3498db",
  "Probably yes" = "#2ecc71",
  "Probably not" = "#e74c3c",
  "Might or might not" = "#f1c40f",
  "Definitely not" = "#9b59b6"
)

# plot scare of AI

AI_scary_data <- data_facility_filtered %>%
  filter(!is.na(att_neg_2)) %>%  
  group_by(att_neg_2) %>%
  summarize(Count = n())

ggplot(AI_scary_data, aes(x = att_neg_2, y = Count, fill = att_neg_2)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = my_colors) +
  labs(
    title = "AI Anxiety",
    x = "I find AI scary",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot benefits of AI

AI_beneficial_data <- data_facility_filtered %>%
  filter(!is.na(att_pos_1)) %>%  
  group_by(att_pos_1) %>%
  summarize(Count = n())

ggplot(AI_beneficial_data, aes(x = att_pos_1, y = Count, fill = att_pos_1)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = my_colors) +
  labs(
    title = "AI's Advantages",
    x = "There are many beneficial applications of AI",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# plots impact 

data_facility_filtered <- data_facility[-1, ]

impact_data <- data_facility_filtered %>%
  filter(!is.na(f_aifm_impact)) %>%  
  group_by(f_aifm_impact) %>%
  summarize(Count = n())

ggplot(impact_data, aes(x = f_aifm_impact, y = Count,fill =f_aifm_impact)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = my_colors_2) +
  labs(
    title = "Impact of Artificial Intelligence on Facility Management",
    x = "Impact",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot AI replace humans

AI_replace_h_data <- data_facility_filtered %>%
  filter(!is.na(att_pos_2)) %>%  
  group_by(att_pos_2) %>%
  summarize(Count = n())

ggplot(AI_replace_h_data, aes(x = att_pos_2, y = Count, fill = att_pos_2)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = my_colors) +
  labs(
    title = "AI vs. Humans: Automating Repetitive Tasks",
    x = "AI systems can replace humans in repetitive tasks",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot AI asist humans in creative tasks

AI_replace_creative_data <- data_facility_filtered %>%
  filter(!is.na(att_pos_3)) %>%  
  group_by(att_pos_3) %>%
  summarize(Count = n())

ggplot(AI_replace_creative_data, aes(x = att_pos_3, y = Count, fill = att_pos_3)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = my_colors) +
  labs(
    title = "AI vs. Humans: Assist in Creative Tasks",
    x = "AI systems can augment/assist humans in creative tasks",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot experience levels

experience_count <- data_facility_filtered %>%
  group_by(demo_experience) %>%
  summarize(Count = n())

ggplot(experience_count, aes(x = demo_experience, y = Count)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Distribution of Experience Levels",
    x = "Experience Level",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# plot importance

# Filter out rows with the question
data_facility_filtered <- data_facility[-1, ]

# Filter out rows with NA 
data_facility_filtered <- data_facility_filtered %>%
  filter(!is.na(f_aifm_importance) & f_aifm_importance != "")

ggplot(data_facility_filtered, aes(x = as.factor(f_aifm_importance), fill = demo_experience)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Importance of AI in FM by Experience Level",
    x = "How important is the use of artificial intelligence in facility management?",
    y = "Count"
  )

# plot AI key roles


data_facility_filtered <- data_facility_filtered %>%
  filter(!is.na(f_aifm_key_roles) & f_aifm_key_roles != "")

# Separate the multiple choices in the "f_aifm_key_roles" column and count the occurrences

key_roles_count <- data_facility_filtered %>%
  separate_rows(f_aifm_key_roles, sep = ",") %>%
  group_by(f_aifm_key_roles) %>%
  summarise(count = n()) %>%
  mutate(f_aifm_key_roles = ifelse(f_aifm_key_roles == "Other (please specify)", "All Aspects of FM", f_aifm_key_roles))

ggplot(key_roles_count, aes(x = f_aifm_key_roles, y = count,fill=f_aifm_key_roles)) +
  geom_bar(stat = "identity") +
  labs(title = "Facility Management and AI: Key Roles and Expectations",
       x = "In what aspects do you think AI will play a key role in facility management?", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot Energy Efficiency in facilities


Energy_data <- data_facility_filtered %>%
  filter(!is.na(f_aifm_energy)) %>%  # Remove rows with NA values
  group_by(f_aifm_energy) %>%
  summarize(Count = n())

ggplot(Energy_data, aes(x = f_aifm_energy, y = Count, fill = f_aifm_energy)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = my_colors_3) +
  labs(
    title = "AI's Impact on Facility Energy Efficiency",
    x = "Do you think AI can help in Energy Efficiency in facilities?",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Linear regresion 


data_facility_filtered$f_aifm_importance_numeric <- as.numeric(factor(data_facility_filtered$f_aifm_importance, levels = c("Slightly important", "Moderately important", "Very important", "Extremely important")))
data_facility_filtered$f_aifm_impact_numeric <- as.numeric(factor(data_facility_filtered$f_aifm_impact, levels = c("Somewhat negative", "Neither positive nor negative", "Somewhat positive", "Extremely positive")))

lm_model <- lm(f_aifm_impact_numeric ~ f_aifm_importance_numeric, data = data_facility_filtered)

summary(lm_model)

# scatter plot with the regression line
ggplot(data_facility_filtered, aes(x = f_aifm_importance_numeric, y = f_aifm_impact_numeric)) +
  geom_point() +                 # Add the points
  geom_smooth(method = "lm") +   # Add the regression line
  labs(x = "Importance (Numeric)", y = "Impact (Numeric)") +
  ggtitle("Linear Regression: Importance vs. Impact")


# Hypothesis testing 

importance_mapping <- c("Slightly important" = 1, "Moderately important" = 2, "Very important" = 3, "Extremely important" = 4)


data_facility_filtered$importance_numeric <- importance_mapping[data_facility_filtered$f_aifm_importance]

data_experience_1 <- data_facility_filtered[data_facility_filtered$demo_experience == "0 - 6 months", ]
data_experience_2 <- data_facility_filtered[data_facility_filtered$demo_experience == "6 - 12 months", ]
data_experience_3 <- data_facility_filtered[data_facility_filtered$demo_experience == "1 - 2 years", ]
data_experience_4 <- data_facility_filtered[data_facility_filtered$demo_experience == "2 - 5 years", ]
data_experience_5 <- data_facility_filtered[data_facility_filtered$demo_experience == "10 - 20 years", ]

anova_result <- aov(importance_numeric ~ demo_experience, data = data_facility_filtered)


summary(anova_result)

p_value <- summary(anova_result)[[1]][["Pr(>F)"]]


print(p_value)


ggplot(data_facility_filtered, aes(x = demo_experience, y = importance_numeric)) +
  geom_boxplot() +
  labs(x = "Experience Level", y = "Importance (Numeric)") +
  ggtitle("Perceptions of AI Importance by Experience Level")

# Calculate the means and confidence intervals
mean_ci <- data_facility_filtered %>%
  group_by(demo_experience) %>%
  summarize(
    mean_importance = mean(importance_numeric),
    ci_lower = mean_importance - 1.96 * (sd(importance_numeric) / sqrt(n())),
    ci_upper = mean_importance + 1.96 * (sd(importance_numeric) / sqrt(n()))
  )


ci_plot <- ggplot(mean_ci, aes(x = demo_experience, y = mean_importance, ymin = ci_lower, ymax = ci_upper)) +
  geom_point(size = 3, color = "blue") +
  geom_errorbar(width = 0.2, color = "blue") +
  labs(x = "Demo Experience", y = "Mean Importance Numeric") +
  ggtitle("Estimates of the Size of the Effect using Confidence Intervals") +
  theme_minimal()

print(ci_plot)

 