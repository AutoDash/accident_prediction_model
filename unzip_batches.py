import sys
import os
from multiprocessing import Pool

npy_format_string = "{:06}"

if __name__ == "__main__":
    # TODO: use argparse
    if len(sys.argv) != 4:
        print("Invalid arguments. Usage: ./unzip_batches [SOURCE DIR] [DEST DIR] [CSV NAME]")
        break

    src_zip_dir, dest_dir, csv_name = sys.argv[1:]
    if not os.path.exists(dest_dir):
        os.system("mkdir -p {}".format(dest_dir))

    zip_paths = [os.path.join(zip_dir, fname) for fname in os.listdir(zip_dir)]

    with open(csv_name.replace('.csv', '') + '.csv') as csv_file:
        csv_writer = csv.writer(csvfile)
        file_count = 1
        for z_path in zip_paths:
            batch_data = np.load(z_path)
            X_batch = batch_data['data']
            Y_batch = batch_data['labels']

            # TODO maybe should parallelize
            for X, Y in zip(X_batch, Y_batch):
                out_file_name = npy_format_string.format(file_count)
                out_path = os.path.join(dest_dir, out_file_name)
                if os.path.exists(out_path):
                    print("File {} already exists".format(outpath))

                csv_writer.writerow([out_file_name] + Y)

                np.save(out_path, X)

                file_count += 1




            

        




