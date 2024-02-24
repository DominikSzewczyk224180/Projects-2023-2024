library(readr)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(stats)
library(pwr)

#Loading the data
text_data <- read_csv("C:/Users/mened/OneDrive/Documents/2023-24a-fai2-adsai-MateyNedyalkov221889/df_text_2023-10-27.csv")
num_data <- read_csv("C:/Users/mened/OneDrive/Documents/2023-24a-fai2-adsai-MateyNedyalkov221889/df_num_2023-10-27.csv")

# File path for saving visualizations
file_path <- "C:/Users/mened/OneDrive/Documents/2023-24a-fai2-adsai-MateyNedyalkov221889/Visuals/"

# Dictionary for renaming some of the variables (for titles)
title_dict <- list(
  "demo_ai_know" = "Participants' knowledge of AI (FM)",
  "ml_dl_famil" = "How familiar are participants with ML and DL? (FM)",
  "f_predmain_curricul" = "Is PM included in the curriculum",
  "f_predmain_impact" = "Does PM have impact on the skill development of students",
  "f_predmain_stuexef" = "Does learning AI predictive maintenance improve FM skills?",
  "f_predmain_ethics" = "How well are ethical considerations integrated into PM at BUas?"
)

# Checking the count of all facility participants
count_facility <- text_data %>% count(demo_domain == "Facility")

# Function for exploring the data
explore_data <- function(data){
  cat("Dimensions:\n")
  print(dim(data))
  
  cat("\nStructure:\n")
  glimpse(data)
  
  cat("\nSummary:\n")
  summary(data)
}

# Function for observing missing data
check_missing_values <- function(data){
  
  any_missing <- anyNA(data)
  
  if (any_missing) {
    cat("There are missing values in the data frame.\n")
  } else {
    cat("There are no missing values in the data frame.\n")
  }
}

# Another function for observing missing data
columns_with_missing_data <- function(data){
  
  missing_columns <- colSums(is.na(data))
  
  cat("Columns with missing values: ")
  cat(names(data)[missing_columns > 0], sep = ", ")
}

explore_data(text_data)

check_missing_values(text_data)
columns_with_missing_data(text_data)


# Removing the unnecessary columns
categories_to_remove <- c('{"ImportId":"QID7"}', "Click to write Choice 10", "What is your domain?")

# Creating graph for number of people in each domain
filtered_data <- text_data %>%
  filter(!(demo_domain %in% categories_to_remove)) %>%
  group_by(demo_domain) %>%
  summarise(Number = n()) %>% 
  filter(!is.na(demo_domain)) %>%
  
  ggplot(aes(x = demo_domain, y = Number, fill = demo_domain)) +
  geom_bar(stat = "identity", width = 0.60) +
  geom_text(aes(label = Number), vjust = 1.6, color = "white", size = 3.5) +
  labs(title = "Number of people in each domain", x = "Domain", y = "Number") + 
  guides(fill = guide_legend(title = "Domains")) +
  theme(legend.position = "none")

ggsave(
  filename = paste0(file_path, "demo_domain_bar_chart.png"),
  plot = filtered_data,
  width = 8,
  height = 4
)

# Filtering to only facility
f_text_data <- text_data %>%
  filter(demo_domain == "Facility") %>%
  select_if(~!all(is.na(.))) %>%        
  filter_all(any_vars(!is.na(.))) 

# Checking demographic data - age and gender of participants of facility
f_gender_age <- f_text_data %>% group_by(demo_gender, demo_age) %>%
                                summarize(Number = n(), .groups = "drop") %>%
                                ggplot(aes(x = demo_gender, y = Number, fill = demo_age)) +
                                geom_bar(stat = "identity", position = "dodge") +
                                labs(title = "Gender and age", x = "Gender", y = "Number") +
                                guides(fill=guide_legend(title="Age groups"))

ggsave(
  filename = paste0(file_path, "gender_age_bar_chart.png"),
  plot = f_gender_age,
  width = 5,
  height = 4
)

# Plotting visuals of pariticipants knowledge of AIs concepts
columns <- c("demo_ai_know", "ml_dl_famil")
plots <- list()
legends <- list()

for (col in columns){
  
  title <- title_dict[[col]]
  
  p <- f_text_data %>%
    filter(!is.na(.data[[col]])) %>%
    group_by(demo_role, .data[[col]]) %>%
    summarize(Number = n(), .groups = "drop") %>%
    ggplot(aes(x = demo_role, y = Number, fill = .data[[col]])) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = title, x = "Role", y = "Number") +
    guides(fill=guide_legend(title="Type of responses")) +
    theme_minimal()
    
  plots[[col]] <- p
  
  legend <- guides(fill = guide_legend(title = "Type of responses"))
  legends[[col]] <- legend
    
    ggsave(paste0(file_path, col, "_bar_chart.png"), p, width = 5, height = 4)
}

combined_figure <- do.call(grid.arrange, plots)

ggsave(paste0(file_path, "AI_knowledge.png"), combined_figure, width = 5, height = 4)


for (col in names(plots)) {
  print(plots[[col]])
}

# Plotting the key components of Facility management 
aifm_key_role <- f_text_data %>%
  select(f_aifm_key_roles) %>%
  separate_rows(f_aifm_key_roles, sep = ",") %>%
  filter(!is.na(f_aifm_key_roles)) %>%
  group_by(f_aifm_key_roles) %>%
  summarise(Count = n()) %>%
  ggplot(aes(x = f_aifm_key_roles, y = Count, fill = f_aifm_key_roles)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = Count), vjust = 1.6, color = "white", size = 3.5) +
  labs(title = "Which component will play an important role in Facility management", x = "Component", y = "Count") +
  guides(fill = guide_legend(title = "Subdomains")) +
  theme(legend.position = "none")

ggsave(
  filename = paste0(file_path, aifm_key_role$f_aifm_key_roles, "key_components.png"),
  plot = aifm_key_role,
  width = 7,
  height = 4
)

# Subsetting to predictive maintenance data
pm_text_data <- f_text_data %>% select(f_predmain_curricul,
                                     f_predmain_impact,
                                     f_predmain_method,
                                     f_predmain_stuexef,
                                     f_predmain_ethics)

pm_text_data <- na.omit(pm_text_data)

# Plotting the predictive maintenance methods
pm_text_methods <- pm_text_data %>% 
                   select(f_predmain_method) %>% 
                   separate_rows(f_predmain_method, sep = ",") %>%
                   group_by(f_predmain_method) %>%
                   summarize(Count = n()) %>%
                   ggplot(aes(x = f_predmain_method, y = Count, fill = f_predmain_method)) +
                   geom_bar(stat = "identity") +
                   labs(title = "The most useful educational methodologies and tools", x = "Methods", y = "Count") +
                   guides(fill=guide_legend(title="Methods")) +
                   theme(legend.position = "none")

ggsave(
  filename = paste0(file_path, "pm_text_methods_bar_chart.png"),
  plot = last_plot(),
  width = 7,
  height = 4
)

# Plotting the variables from the survey
column_names <- colnames(pm_text_data)
exclude_column <- "f_predmain_method"

plots <- list()

for (col in column_names) {
  if (col != exclude_column) {
    
    title <- title_dict[[col]]
    
    p <- pm_text_data %>%
      group_by(.data[[col]]) %>%
      summarize(Count = n()) %>%
      ggplot(aes(x = .data[[col]], y = Count, fill = .data[[col]])) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = ""), vjust = -0.5) +
      labs(title = title) +
      theme(legend.position = "none")
      theme_minimal() 
    
    plots[[col]] <- p
    
    ggsave(paste0(file_path, col, "_bar_chart.png"), p, width = 7, height = 4)
  }
}


for (col in names(plots)) {
  print(plots[[col]])
}

combined_figure <- do.call(grid.arrange, plots)

ggsave(paste0(file_path, "PM_answers.png"), combined_figure, width = 12, height = 7)

# Subsetting numerical data for the t-test and linear regression
pm_num_data <- num_data %>% select(f_predmain_curricul,
                                       f_predmain_impact,
                                       f_predmain_method,
                                       f_predmain_stuexef,
                                       f_predmain_ethics)

# Converting columns to numerical
columns_to_convert <- c("f_predmain_stuexef", "f_predmain_impact", "f_predmain_curricul")

pm_num_data <- pm_num_data %>%
  mutate_at(vars(columns_to_convert), as.numeric)

pm_num_data <- pm_num_data[-c(1, 2),]

pm_num_data <- na.omit(pm_num_data)

pm_num_methods <- pm_num_data %>% select(f_predmain_method) %>%  
  separate_rows(f_predmain_method, sep = ",")

n <- 10
alpha <- 0.05
power <- 0.8

power_analys <- pwr.t.test(sig.level = alpha, power = power, n = n, type = "one.sample")

# Performing one sample t-test
hypothesized_value <- 4

t_test_result <- t.test(pm_num_data$f_predmain_impact, mu = hypothesized_value, alt="less")

x <- seq(-3, 3, length.out = 1000)

y <- dt(x, df = length(pm_num_data) - 1)

t_dist_df <- data.frame(x, y)

# Visualizing t-distribution
test_graph <- ggplot(t_dist_df, aes(x = x, y = y)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_vline(xintercept = t_test_result$statistic, color = "red", linetype = "dashed", size = 1) +
  geom_area(data = subset(t_dist_df, x >= t_test_result$statistic), aes(x = x, y = y), fill = "gray", alpha = 0.5) +
  labs(
    title = "T-Distribution with One-Sample T-Test",
    x = "T-Statistic",
    y = "Probability Density"
  ) +
  theme_minimal()

ggsave(paste0(file_path, "One-sample_T-Test.png"), test_graph, width = 5, height = 4)

#ci <- t_test_result$conf.int

#df <- data.frame(mean = mean(data), lower = ci[1], upper = ci[2])

#test_graph <- test_graph +
  #geom_ribbon(data = subset(t_dist_df, x >= t_test_result$statistic),
              #aes(x = x, ymin = df$lower, ymax = df$upper),
              #fill = "blue", alpha = 0.3)

# Checking the correlation of the data
corr <- pm_num_data %>% summarize(cor(f_predmain_stuexef, f_predmain_impact))

# Performing linear regression
model <- lm(f_predmain_stuexef ~ f_predmain_impact, data = pm_num_data)

# Plotting a scatter plot with linear line
linear_graph <- ggplot(pm_num_data, aes(x = f_predmain_impact, y = f_predmain_stuexef)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE) +
  labs(x = "Exposure to AI Techniques", y = "Ability in Practical Scenarios")

ggsave(paste0(file_path, "Linear_graph.png"), linear_graph, width = 5, height = 4)

qqnorm(model$residuals)
qqline(model$residuals)
