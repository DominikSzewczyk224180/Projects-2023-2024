install.packages('psych')

#loading libraries

library(readr)
library(ggplot2)
library(tidyverse)
library(stringr)
library(dplyr)



#Reading CSV files
setwd("C:\\Users\\mani1\\Documents\\GitHub\\2023-24a-fai2-adsai-ImaniJamirSenior225619\\Data")

security_data <- read.csv("C:\\Users\\mani1\\Documents\\GitHub\\2023-24a-fai2-adsai-ImaniJamirSenior225619\\Data\\df_sec_text.csv")
security_num_data <- read.csv("C:\\Users\\mani1\\Documents\\GitHub\\2023-24a-fai2-adsai-ImaniJamirSenior225619\\Data\\df_sec_num.csv")


# Research Question 1: To what extent do AI systems in facilities impact data security?
grouped_data_impact <- security_data %>%
  group_by(demo_role, demo_age, f_sec_help) %>%
  summarise(count = n())

customLikertScale <- c("Probably yes", "Definitely yes")
customColors <- c("Probably yes" = "purple", "Definitely yes" = "orange")

ggplot(grouped_data_impact, aes(x = demo_role, y = count, fill = factor (f_sec_help, levels = customLikertScale))) +
  geom_bar(stat = "identity", position = position_dodge(preserve = "single"), color = "black") +
  labs(title = "AI Systems Impact on Data Security",
       x = "Role",
       y = "values") +
  scale_y_continuous() +
  facet_wrap(~demo_age, scales = "free_x") +
  theme_minimal() +
  theme(legend.title = element_text("Impact Level"),
        legend.position = "right",
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 0.5, size = 10),  # Adjust x-axis label appearance
        axis.text.y = element_text(size = 10),
        axis.title.x = element_text(size = 12),  
        axis.title.y = element_text(size = 12),  
        panel.grid.major.y = element_line(color = "black"),  
        panel.grid.minor = element_blank()) + # Remove grid lines
scale_fill_manual(values = rainbow(length(customLikertScale)))
# Research Question 2: How do students and teachers perceive the implementation of AI in facilities management at buas in terms of data security and privacy concerns?
grouped_data_concern <- security_data %>%
  group_by(demo_role, demo_age, f_sec_concern) %>%
  summarise(count = n())


customLikertScale <- c('Not at all concerned',"Slightly concerned", "Moderately concerned", "Somewhat concerned", "Extremely concerned")

ggplot(grouped_data_concern, aes(x = demo_role, y = count, fill = factor(f_sec_concern, levels = customLikertScale))) +
  geom_bar(stat = "identity", position = position_dodge(preserve = "single"), color = "black") +
  labs(title = "Perceived Level of Concerns by Role and Age",
       x = "Role",
       y = "Values") +
  scale_y_continuous() + # Set fill scale explicitly
  facet_wrap(~demo_age, scales = "free_x") +
  theme_minimal() +
  theme(legend.title = element_blank(),
        legend.position = "right",
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 0.5, size = 10),  # Adjust x-axis label appearance
        axis.text.y = element_text(size = 10),
        axis.title.x = element_text(size = 12),  
        axis.title.y = element_text(size = 12),  
        panel.grid.major.y = element_line(color = "black"),  # Add horizontal measure line
        panel.grid.minor = element_blank()) +# Removing grid lines
  scale_fill_manual(values = rainbow(length(customLikertScale)))
# Research Question 3:  What are the key factors that influence the attitudes of students and teachers toward AI implementation in facilities management at BUAS, specifically in terms of the confidence level, and how do these factors contribute to either positive or negative perceptions?"
# Group and summarize the data by f_sec_confidence, demo_age, and demo_role
grouped_data_confidence <- security_data %>%
  group_by(f_sec_confidence, demo_age, demo_role) %>%
  summarise(count = n())

# Define the custom Likert scale order
customLikertScale <- c('Not confident', 'Neutral', 'Somewhat confident', 'Very confident')

ggplot(grouped_data_confidence, aes(x = demo_role, y = count, fill = factor(f_sec_confidence, levels = customLikertScale))) +
  geom_bar(stat = "identity", position = position_dodge(preserve = "single"), color = 'black') +
  scale_y_continuous() +
  labs(title = "Perceived attitude by Confidence Level, Age, and Role",
       x = "role",
       y = "Values") +
  facet_wrap(~demo_age, scales = "free_x") +
  theme_minimal() +
  theme(legend.title = element_text("Confidence Level"), 
        legend.position = "right",
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 0.5, size =10),
        axis.text.y = element_text(size = 10),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        panel.grid.major.y = element_line(color = 'black'),
        panel.grid.minor = element_blank()) +
  scale_fill_manual(values = rainbow(length(customLikertScale)))


library(dplyr)
library(ggplot2)

# Define a custom color palette for each row
custom_colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")

# Select the most and least influential factors
selected_factors <- security_data %>%
  select(demo_role, demo_age, f_sec_influence_1, f_sec_influence_8)

# Reshape the data to a longer format
reshaped_data <- selected_factors %>%
  pivot_longer(cols = c(f_sec_influence_1, f_sec_influence_8), names_to = "Influence", values_to = "Rank")

# Create a grouped bar chart with custom colors
ggplot(reshaped_data, aes(x = demo_role, y = Rank, fill = Influence)) +
  geom_bar(stat = "identity", position = position_dodge(preserve = "single"), color = 'black') +
  labs(title = "Most and Least Influential Factors on AI Implementation by Role and Age",
       x = "Role",
       y = "Rank") +
  scale_fill_manual(values = custom_colors) + 
  facet_wrap(~demo_age, scales = "free_x") +
  theme_minimal() +
  theme(legend.title = element_blank(),
        legend.position = "right",
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 0.5, size = 10),
        axis.text.y = element_text(size = 10),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        panel.grid.major.y = element_line(color = 'black'),
        panel.grid.minor = element_blank())


# Select only the most and least influential factors
selected_factors <- security_data %>%
  select(demo_role, demo_age, f_sec_influence_1, f_sec_influence_8)

# Find the most common ranking in the most influential column (f_sec_influence_1)
most_influential_common <- selected_factors %>%
  summarise(most_common_influence = f_sec_influence_1)

# Find the most common ranking in the least influential column (f_sec_influence_8)
least_influential_common <- selected_factors %>%
  summarise(least_common_influence = f_sec_influence_8)

# Creating a data frame with the results
common_rankings <- data.frame(Factor = c("Most Influential", "Least Influential"),
                              Ranking = c(most_influential_common$most_common_influence, 
                                          least_influential_common$least_common_influence))

# Print table
common_rankings






# Performing a t-test for the f_sec_concern variable
t_test_result <- t.test(security_num_data$f_sec_concern[security_num_data$demo_role == 1], 
                        security_num_data$f_sec_concern[security_num_data$demo_role == 2])

# Print the t-test result
print(t_test_result)



# Performing a t-test for the f_sec_concern variable
t_test_result <- t.test(security_num_data$f_sec_confidence[security_num_data$demo_role == 1], 
                        security_num_data$f_sec_confidence[security_num_data$demo_role == 2])

# Print the t-test result
print(t_test_result)



# Create a contingency table for your data
table_security <- table(security_data$demo_role, security_data$f_sec_confidence)

# Perform the chi-squared test
chi_squared_security <- chisq.test(table_security)

# Display the chi-squared test result
print(chi_squared_security)

# Define example colors for demo_role and f_sec_confidence
role_colors <- c("purple", "orange") # Adjust these colors
confidence_colors <- c("cyan", "magenta", "yellow") # Adjust these colors

# Create a barplot with explicit colors
barplot(as.matrix(table_security), beside = TRUE, 
        col = matrix(c(rep(role_colors, each = length(confidence_colors)), length.out = nlevels(security_data$demo_role)),
                     names.arg = unique(security_data$demo_role))
        legend("topright", legend = unique(security_data$demo_role), 
               fill = role_colors,
               cex = 0.8, bty = "n", xpd = NA, x.intersp = 1.5)

        

        # Convert the likert scale of 'f_sec_help' to numeric
        # Create the f_sec_help_numeric variable based on the likert scale
        security_data$f_sec_help_numeric <- factor(security_data$f_sec_help, 
                                                   levels = c('Definitely not', 'Probably not', 'Might or might not', 'Probably yes', 'Definitely yes'), 
                                                   labels = c(1, 2, 3, 4, 5))
        
        # Converting the f_sec_help_numeric variable to numeric
        security_data$f_sec_help_numeric <- as.numeric(as.character(security_data$f_sec_help_numeric))
        
        
        # Filter data for 'Student' and 'Educator' roles
        filtered_data <- security_data %>% filter(demo_role %in% c("Student", "Educator"))
        
        # Create a linear regression model
        model <- lm(f_sec_help_numeric ~ demo_role, data = filtered_data)
        
        # Print the summary of the model
        summary(model)
        
        # Predicting values using the model
        filtered_data$predicted_values <- predict(model, newdata = filtered_data)
        
        # Creating a ggplot to visualize the model
        ggplot(filtered_data, aes(x = demo_role, y = f_sec_help_numeric, group = demo_role)) +
          # Adding jittered points
          geom_jitter(width = 0.2, aes(color = demo_role), alpha = 0.6) +
          # Adding box plots
          geom_boxplot(aes(fill = demo_role), width = 0.3, alpha = 0.4, position = position_dodge(width = 0.8)) +
          # Adding predicted means (as black X's)
          geom_point(aes(y = predicted_values), color = "black", shape = 4) +
          # Showing residuals as line segments
          geom_segment(aes(xend = demo_role, yend = predicted_values), color = "gray50") +
          labs(title = "Impact of AI on Security by Role",
               x = "Role", y = "Security Impact Score") +
          theme_minimal() +
          scale_color_manual(values = c('gold', "purple")) +
          scale_fill_manual(values = c("yellow", "purple"))
        


        
        
        






