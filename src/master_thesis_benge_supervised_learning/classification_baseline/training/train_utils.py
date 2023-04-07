import wandb

class TrainUtils:

    def caluculate_and_log_accuracy_per_class_training(accuracy, number_of_samples):
        for i in range(len(accuracy)):
            wandb.log({"Accuracy class training" + str(i): accuracy[i] / number_of_samples})

    def caluculate_and_log_accuracy_per_class_validation(accuracy, number_of_samples):
        for i in range(len(accuracy)):
            wandb.log({"Accuracy class validation" + str(i): accuracy[i] / number_of_samples})

    def caluculate_and_log_f1_per_class_training(accuracy, number_of_samples):
        for i in range(len(accuracy)):
            wandb.log({"F1 class training" + str(i): accuracy[i] / number_of_samples})

    def caluculate_and_log_f1_per_class_validation(accuracy, number_of_samples):
        for i in range(len(accuracy)):
            wandb.log({"F1 class validation" + str(i): accuracy[i] / number_of_samples })