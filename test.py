import os
import torch
import socket
from pytt.utils import seed_state, set_random_state, read_pickle, write_pickle
from pytt.email import EmailSender, default_onerror, check_attachment_error
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.distributed import distributed_wrapper
from pytt.testing.tester import Tester
from pytt.logger import logger
from processing.dataset import init_dataset
from hierarchy import Hierarchy
#from fairseq.legacy_distributed_data_parallel\
#        import LegacyDistributedDataParallel as LDDP
from model_loader import load_model_components
from argparse import ArgumentParser
import parameters as p


def main(model_type, val_file, checkpoint_folder, hierarchy, supervised=False, device='cuda:0', batch_size=p.batch_size, limit_rows_val=p.limit_rows_val, subbatches=p.subbatches, num_workers=p.num_workers, email_sender=None, results_folder=None, noload=False):
    if checkpoint_folder is None or noload:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    val_dataset = init_dataset(val_file, limit_rows=limit_rows_val)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size)
    model_file = os.path.join(checkpoint_folder, 'model_state.tpkl') if not noload else None
    batcher, model, postprocessor = load_model_components(model_type, hierarchy, run_type='testing', device=device, model_file=model_file, cluster=supervised)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=subbatches, num_workers=num_workers)
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    tester = Tester(model, postprocessor, val_iterator)
#    tester = Tester(model, postprocessor, val_iterator, tensorboard_dir=os.path.join(load_checkpoint_folder, 'tensorboard/test'))
    if results_folder is None:
        results_folder = os.path.join(checkpoint_folder, 'results')
    os.mkdir(results_folder)
    postprocessor.add_output_dir(results_folder)
    total_output_batch = tester.test()
    with open(os.path.join(results_folder, 'scores.txt'), 'w') as f:
        f.write(str(total_output_batch))
    #total_output_batch.write_results()
    #write_pickle(postprocessor.summary_stats, os.path.join(checkpoint_folder, 'summary_stats.pkl'))
    if email_sender is not None:
        def onerror(e):
            if check_attachment_error(e):
                logger.log("Trying to send without attachment")
                email_sender.send_email(str(total_output_batch))
            else:
                default_onerror(e)
        attachments = postprocessor.get_summary_attachment_generator()
        email_sender.send_email("Testing is done!\n\n"+str(total_output_batch), attachments=attachments, onerror=onerror)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("checkpoint_folder")
    parser.add_argument("-n", "--noload", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-e", "--email", action="store_true")
    parser.add_argument("--sender_password", default=None)
    parser.add_argument("-s", "--supervised", action="store_true")
    parser.add_argument("--supervised_data_dir", default=None)
    parser.add_argument("--results_folder", default=None)
    args = parser.parse_args()

    val_file = os.path.join(p.data_dir, 'val.data') if not args.supervised else os.path.join(args.supervised_data_dir, 'supervised.data')

    if args.email:
        email_sender = EmailSender(smtp_server=p.smtp_server, port=p.port, sender_email=p.sender_email, sender_password=args.sender_password, receiver_email=p.receiver_email, subject="%s: testing %s model" % (socket.gethostname(), args.model_type))
        email_sender.send_email("Starting to test %s model." % args.model_type)
    else:
        email_sender = None

    if args.supervised:
        hierarchy_file = os.path.join(args.supervised_data_dir, 'hierarchy.pkl')
        if not os.path.exists(hierarchy_file):
            hierarchy_file = os.path.join(args.checkpoint_folder, 'hierarchy.pkl')
    else:
        hierarchy_file = os.path.join(args.checkpoint_folder, 'hierarchy.pkl')

    hierarchy = Hierarchy.from_dict(read_pickle(hierarchy_file))

    try:
        main(args.model_type, val_file, args.checkpoint_folder, hierarchy, supervised=args.supervised, device=args.device, email_sender=email_sender, results_folder=args.results_folder, noload=args.noload)
#        nprocs = 2
#        main_distributed = distributed_wrapper(main, nprocs)
#        main_distributed(args.model_type, val_file, args.checkpoint_folder, device=args.device)
    except Exception as e:
        if email_sender is not None:
            email_sender("Got an exception:\n%s" % e)
        raise e
