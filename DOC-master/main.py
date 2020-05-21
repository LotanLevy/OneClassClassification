
import utils
import configurations
import dataloader
from os import listdir
from os.path import isfile, join
import numpy as np
from Networks.traintest import Trainer
import tensorflow as tf
from Losses import compactnes_loss, descriptiveness_loss



def get_target_images_by_classes(dir, classes_names):
    paths = []
    for sub_dir in listdir(dir):
        if sub_dir in classes_names:
            class_path = join(dir, sub_dir)
            for sub_class in listdir(class_path):
                print(join(class_path, sub_class))
                paths += dataloader.get_images_path(join(class_path, sub_class))
    return paths




def train(ref_dataloader, target_dataloader, trainer, batchs_num, iterations_for_epochs, epochs):
    trainstep = trainer.get_step()
    for i in range(int(iterations_for_epochs * epochs)):
        ref_batch_x, ref_batch_y = ref_dataloader.read_images_batch(batchs_num)
        target_batch_x, _ = target_dataloader.read_images_batch(batchs_num)
        trainstep(ref_batch_x, ref_batch_y, target_batch_x)
        print("loss after {} iterations: D_loss {}, c_loss {}, total loss {}".format(i + 1,
                                                                   trainer.D_loss_logger.result(),
                                                                   trainer.C_loss_logger.result(),
                                                                                     trainer.D_loss_logger.result() +
                                                                                     trainer.loss_lambda *
                                                                                     trainer.C_loss_logger.result()))






def main():
    args = configurations.get_args()
    ref_labels = dataloader.read_labels_file(args.reflabelpath)
    classes_num = len(np.unique(ref_labels))
    ref_images_paths = dataloader.get_images_path(args.refpath)
    target_images_paths = get_target_images_by_classes(args.targetpath, ["knife", "sword"])
    ref_dataloader = dataloader.Dataloader(ref_images_paths,classes_num, ref_labels)
    target_dataloader = dataloader.Dataloader(target_images_paths,classes_num )
    network = utils.get_network(args.nntype)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    trainer = Trainer(network, optimizer, args.lambd, compactnes_loss, descriptiveness_loss)

    num_iterations = max(len(ref_images_paths) / args.batches, 1)

    train(ref_dataloader, target_dataloader, trainer, args.batches, num_iterations, args.epochs)





if __name__ == "__main__":
    main()
