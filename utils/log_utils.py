import os
def check_folder(path):
    if not os.path.exists(os.path.join(args.log, now)):
        os.makedirs(log_dir)
        
def display(log_file, msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()