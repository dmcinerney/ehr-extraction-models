import os
import torch
import socket
from pytt.utils import seed_state, set_random_state, read_pickle, write_pickle
from pytt.email import EmailSender
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.distributed import distributed_wrapper
from pytt.testing.tester import Tester
from pytt.logger import logger
from processing.dataset import init_dataset
#from fairseq.legacy_distributed_data_parallel\
#        import LegacyDistributedDataParallel as LDDP
from model_loader import load_model_components
from argparse import ArgumentParser
import parameters as p
from hierarchy import Hierarchy


def main(model_type, val_file, checkpoint_folder, hierarchy, supervised=False, device='cuda:0', batch_size=p.batch_size, subbatches=p.subbatches, num_workers=p.num_workers, email_sender=None):
    if checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    val_dataset = init_dataset(val_file)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size)
    model_file = os.path.join(checkpoint_folder, 'model_state.tpkl')
    batcher, model, postprocessor = load_model_components(model_type, hierarchy, run_type='testing', device=device, model_file=model_file, cluster=supervised)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=subbatches, num_workers=num_workers)
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    tester = Tester(model, postprocessor, val_iterator)
#    tester = Tester(model, postprocessor, val_iterator, tensorboard_dir=os.path.join(load_checkpoint_folder, 'tensorboard/test'))
    postprocessor.add_output_dir(checkpoint_folder)
    total_output_batch = tester.test()
    with open(os.path.join(checkpoint_folder, 'scores.txt'), 'w') as f:
        f.write(str(total_output_batch))
    #total_output_batch.write_results()
    #write_pickle(postprocessor.summary_stats, os.path.join(checkpoint_folder, 'summary_stats.pkl'))
    if email_sender is not None:
        attachments = postprocessor.get_summary_attachment_generator()
        email_sender.send_email("Testing is done!\n\n"+str(total_output_batch), attachments=attachments)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("checkpoint_folder")
    parser.add_argument("--data_file", default=os.path.join(p.data_dir, 'val.data'))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-e", "--email", action="store_true")
    parser.add_argument("--sender_password", default=None)
    parser.add_argument("-s", "--supervised", action="store_true")
    parser.add_argument("--hierarchy", default=None)
    args = parser.parse_args()

    if args.email:
        email_sender = EmailSender(smtp_server=p.smtp_server, port=p.port, sender_email=p.sender_email, sender_password=args.sender_password, receiver_email=p.receiver_email, subject="%s: testing %s model" % (socket.gethostname(), args.model_type))
        email_sender.send_email("Starting to test %s model." % args.model_type)
    else:
        email_sender = None

    hierarchy = Hierarchy.from_dict(read_pickle(args.hierarchy))\
                if args.hierarchy is not None else\
                os.path.join(checkpoint_folder, 'hierarchy.pkl')

    main(args.model_type, args.data_file, args.checkpoint_folder, hierarchy, supervised=args.supervised, device=args.device, email_sender=email_sender)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(args.model_type, val_file, args.checkpoint_folder, device=args.device)
