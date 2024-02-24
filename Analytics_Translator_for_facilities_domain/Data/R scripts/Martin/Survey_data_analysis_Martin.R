
library(readr)
library(ggplot2)
library(tidyverse)


#Loaded libraries


survey_data <- read.csv("C:/Users/MSI/Desktop/Y2BA/df_space_opt_text.csv")


grouped_data_belief <- survey_data %>%
  group_by(demo_role,demo_age, f_so_belief) %>%
  count()

#Comparing belief based on age and role

ggplot(grouped_data_belief, aes(x = demo_role, y = n, fill = f_so_belief)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~demo_age, scales = "free_x") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Do you believe AI can help in Space Optimization within facilities ?", y = "Count")


grouped_data_difficulty <- survey_data %>%
  group_by(demo_role, demo_age, f_so_difficulty) %>%
  count()

#Comparing difficulty of implementation based on age and role

title_text_adp <- "Do you think it would be difficult to implement AI for Space Optimization in facilities?"
wrapped_title <- stringr::str_wrap(title_text_adp, width = 60)

ggplot(grouped_data_difficulty, aes(x = demo_role, y = n, fill = f_so_difficulty)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~demo_age, scales = "free_x") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(size = 12)  
  ) +
  labs(title = wrapped_title, y = "Count")

#Comparing ai adoption based on age and role

grouped_data_adoption <- survey_data %>%
  group_by(demo_role, demo_age, f_so_adoption) %>%
  count()



title_text_adp <- "How likely do you think it is that AI will become a standard tool for space optimization in facility management in the next decade?"
wrapped_title <- stringr::str_wrap(title_text_adp, width = 60)


ggplot(grouped_data_adoption, aes(x = demo_role, y = n, fill = f_so_adoption)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~demo_age, scales = "free_x") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(size = 12)  
  ) +
  labs(title = wrapped_title, y = "Count")

#Comparing ethical implications based on age and role

grouped_data_ethical <- survey_data %>%
  group_by(demo_role, demo_age, f_so_ethical) %>%
  count()



title_text_eth <- "Do you think there are ethical implications in implementing AI for Space Optimization?"
wrapped_title <- stringr::str_wrap(title_text_eth, width = 60)  

ggplot(grouped_data_ethical, aes(x = demo_role, y = n, fill = f_so_ethical)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~demo_age, scales = "free_x") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(size = 12) 
  ) +
  labs(title = wrapped_title, y = "Count")



table_ethical <- table(survey_data$demo_role[survey_data$demo_role %in% c("Educator", "Student")], survey_data$f_so_ethical[survey_data$demo_role %in% c("Educator", "Student")])

#Null Hypothesis: Students and Lecturers do not have differing opinions when it comes to the ethical implications of using AI for space optimization
#Alternative Hypothesis: There is a difference in opinions between students and lecturers when it comes to the ethical implications of using AI for space optimization

chi_squared_ethical <- chisq.test(table_ethical)
chi_squared_ethical
barplot(as.matrix(table_ethical), beside=TRUE, col=c("red", "blue"))
legend("top", legend=c("Educator", "Student"), fill=c("red", "blue"), 
       cex=0.8, bty="n", xpd=NA, x.intersp=1.5)


#p-value lower than 0.05 indicates we can reject the Null Hypothesis


survey_data$f_so_belief_numeric <- factor(survey_data$f_so_belief, levels=c("Definitely not", "Probably not", "Might or might not", "Proabably yes", "Definitely yes"), labels=c(1, 2, 3, 4, 5))
survey_data$f_so_belief_numeric <- as.numeric(as.character(survey_data$f_so_belief_numeric))



filtered_data <- survey_data %>% filter(demo_role %in% c("Student", "Educator"))



model <- lm(f_so_belief_numeric ~ demo_role, data=filtered_data)



summary(model)

filtered_data$predicted_values <- predict(model, newdata = filtered_data)



ggplot(filtered_data, aes(x = demo_role, y = f_so_belief_numeric, group = demo_role)) +
  
  # Adding jittered points
  geom_jitter(width = 0.2, aes(color = demo_role), alpha = 0.6) +
  
  # Adding box plots
  geom_boxplot(aes(fill = demo_role), width = 0.3, alpha = 0.4, position = position_dodge(width = 0.8)) +
  
  # Adding predicted means (as black X's)
  geom_point(aes(y = predicted_values), color = "black", shape = 4) +  
  
  # Showing residuals as line segments
  geom_segment(aes(xend = demo_role, yend = predicted_values), color = "gray50") +
  
  labs(title = "Belief in Space Optimization AI by Role",
       x = "Role", y = "Belief Score") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "red")) +
  scale_fill_manual(values = c("blue", "red"))
