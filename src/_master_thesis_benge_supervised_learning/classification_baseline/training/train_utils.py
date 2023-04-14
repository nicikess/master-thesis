import wandb


class TrainUtils:
    def caluculate_and_log_accuracy_per_class_training(accuracy):
        for i in range(len(accuracy)):
            wandb.log({"Accuracy class training " + str(i): accuracy[i]})

    def caluculate_and_log_accuracy_per_class_validation(accuracy):
        for i in range(len(accuracy)):
            wandb.log({"Accuracy class validation " + str(i): accuracy[i]})

    def caluculate_and_log_f1_per_class_training(accuracy):
        for i in range(len(accuracy)):
            wandb.log({"F1 class training " + str(i): accuracy[i]})

    def caluculate_and_log_f1_per_class_validation(accuracy):
        for i in range(len(accuracy)):
            wandb.log({"F1 class validation " + str(i): accuracy[i]})
