import os
import socket
import subprocess
from shutil import copyfile
import torch
from pytt.utils import seed_state, set_random_state, read_pickle, write_pickle
from pytt.email import EmailSender
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.testing.tester import Tester
from tracker import Tracker
from pytt.distributed import distributed_wrapper
from pytt.logger import logger
from processing.dataset import init_dataset
from hierarchy import Hierarchy
#from fairseq.legacy_distributed_data_parallel\
#        import LegacyDistributedDataParallel as LDDP
from model_loader import load_model_components
from shutil import copyfile
from argparse import ArgumentParser
import parameters as p

class SupervisedTestingFunc:
    def __init__(self, val_file, training_model, model_type, hierarchy, device, batch_size, subbatches, num_workers, results_folder, email_sender):
        self.val_dataset = init_dataset(val_file)
        self.training_model = training_model
        self.batcher, self.model, self.postprocessor = load_model_components(model_type, hierarchy, run_type='testing', device=device, cluster=True)
        subprocess.run(["mkdir", results_folder])
        self.batch_size, self.subbatches, self.num_workers = batch_size, subbatches, num_workers
        self.results_folder = results_folder
        self.email_sender = email_sender

    def __call__(self, iteration_info):
        logger.log("Running Supervised Testing")
        self.model.load_state_dict(self.training_model.state_dict())
        results_folder = os.path.join(self.results_folder, 'results_%s' % iteration_info.iterator_info.batches_seen)
        os.mkdir(results_folder)
        self.postprocessor.add_output_dir(results_folder)
        val_indices_iterator = init_indices_iterator(len(self.val_dataset), self.batch_size)
        val_iterator = self.batcher.batch_iterator(self.val_dataset, val_indices_iterator, subbatches=self.subbatches, num_workers=self.num_workers)
        tester = Tester(self.model, self.postprocessor, val_iterator)
        total_output_batch = tester.test()
        with open(os.path.join(results_folder, 'scores.txt'), 'w') as f:
            f.write(str(total_output_batch))
        if self.email_sender is not None:
            attachments = self.postprocessor.get_summary_attachment_generator()
            self.email_sender("Testing is done!\n\n"+str(total_output_batch), attachments=attachments)
        logger.log("Testing is done!")

def main(model_type, train_file, hierarchy, counts_file, val_file=None, save_checkpoint_folder=None, load_checkpoint_folder=None, device='cuda:0',
         batch_size=p.batch_size, epochs=p.epochs, limit_rows_train=p.limit_rows_train, limit_rows_val=p.limit_rows_val, subbatches=p.subbatches,
         num_workers=p.num_workers, checkpoint_every=p.checkpoint_every, copy_checkpoint_every=p.copy_checkpoint_every, val_every=p.val_every,
         email_every=None, email_sender=None, expensive_val_every=None, supervised_val_file=None, supervised_val_hierarchy=None,
         results_folder=None):
    if load_checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(load_checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    train_dataset = init_dataset(train_file, limit_rows=limit_rows_train)
    if val_file is not None:
        val_dataset = init_dataset(val_file, limit_rows=limit_rows_val)
    if load_checkpoint_folder is None:
        indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=epochs)
        if val_file is not None:
            val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
        model_file, optimizer_file = None, None
    else:
        indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'train_indices_iterator.pkl'))
        indices_iterator.set_stop(epochs=epochs)
        if val_file is not None:
            val_indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'val_indices_iterator.pkl'))
            val_indices_iterator.set_stop(iterations=len(indices_iterator))
        model_file, optimizer_file = os.path.join(load_checkpoint_folder, 'model_state.tpkl'), os.path.join(load_checkpoint_folder, 'optimizer_state.tpkl')
        if not os.path.exists(optimizer_file):
            optimizer_file = None
    batcher, model, postprocessor, optimizer = load_model_components(model_type, hierarchy, device=device, model_file=model_file,
                                                                     optimizer_file=optimizer_file, counts_file=counts_file)
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=subbatches, num_workers=num_workers)
    if val_file is not None:
        val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=subbatches)
    else:
        val_iterator = None
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    expensive_val_func = SupervisedTestingFunc(supervised_val_file, model, model_type, supervised_val_hierarchy, device, batch_size, subbatches, num_workers, results_folder, email_sender)\
                         if expensive_val_every is not None else None
    tracker = Tracker(checkpoint_folder=save_checkpoint_folder, checkpoint_every=checkpoint_every, copy_checkpoint_every=copy_checkpoint_every,
                      email_every=email_every, email_sender=email_sender, expensive_val_every=expensive_val_every, expensive_val_func=expensive_val_func)
#    if load_checkpoint_folder is not None:
#        tracker.needs_graph = False
    tracker.needs_graph = False
    trainer = Trainer(model, postprocessor, optimizer, batch_iterator, val_iterator=val_iterator, val_every=val_every, tracker=tracker)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("--data_dir", default=p.data_dir)
    parser.add_argument("--code_graph_file", default=p.code_graph_file)
    parser.add_argument("--save_checkpoint_folder", default=None)
    parser.add_argument("--load_checkpoint_folder", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-e", "--email", action="store_true")
    parser.add_argument("--sender_password", default=None)
    parser.add_argument("--expensive_val_every", default=p.expensive_val_every)
    parser.add_argument("--supervised_data_dir", default=None)
    parser.add_argument("--results_folder", default=None)
    args = parser.parse_args()

    if args.email:
        email_sender = EmailSender(smtp_server=p.smtp_server, port=p.port, sender_email=p.sender_email, sender_password=args.sender_password, receiver_email=p.receiver_email, subject="%s: training %s model" % (socket.gethostname(), args.model_type))
        email_sender.send_email("Starting to train %s model." % args.model_type)
        email_every = p.email_every
    else:
        email_sender = None
        email_every = None

    train_file = os.path.join(args.data_dir, 'train.data')
    val_file = os.path.join(args.data_dir, 'val.data')
    counts_file = os.path.join(args.data_dir, 'counts.pkl')
    used_targets_file = os.path.join(args.data_dir, 'used_targets.txt')

    hierarchy = Hierarchy.from_graph(read_pickle(args.code_graph_file))

    if args.save_checkpoint_folder is not None:
        write_pickle(hierarchy.to_dict(), os.path.join(args.save_checkpoint_folder, 'hierarchy.pkl'))
        if os.path.exists(counts_file):
            copyfile(counts_file, os.path.join(args.save_checkpoint_folder, 'counts.pkl'))
        if os.path.exists(used_targets_file):
            copyfile(used_targets_file, os.path.join(args.save_checkpoint_folder, 'used_targets.txt'))

    if args.expensive_val_every is not None:
        supervised_val_file = os.path.join(args.supervised_data_dir, 'supervised.data')
        hierarchy_file = os.path.join(args.supervised_data_dir, 'hierarchy.pkl')
        supervised_val_hierarchy = Hierarchy.from_dict(read_pickle(hierarchy_file))\
                                   if os.path.exists(hierarchy_file) else hierarchy
    else:
        supervised_val_file = None
        supervised_val_hierarchy = None

    expensive_val_every = int(args.expensive_val_every) if args.expensive_val_every is not None else None
    try:
        main(args.model_type, train_file, hierarchy, counts_file, val_file=val_file,
             save_checkpoint_folder=args.save_checkpoint_folder, load_checkpoint_folder=args.load_checkpoint_folder,
             device=args.device, email_every=email_every, email_sender=email_sender, expensive_val_every=expensive_val_every,
             supervised_val_file=supervised_val_file, supervised_val_hierarchy=supervised_val_hierarchy,
             results_folder=args.results_folder)
#        nprocs = 2
#        main_distributed = distributed_wrapper(main, nprocs)
#        main_distributed(args.model_type, train_file, args.code_graph_file, val_file=val_file,
#             save_checkpoint_folder=args.save_checkpoint_folder, load_checkpoint_folder=args.load_checkpoint_folder,
#             device=args.device, email_every=email_every, email_sender=email_sender)
    except Exception as e:
        if email_sender is not None:
            email_sender("Got an exception:\n%s" % e)
        raise e
