import os
import csv


def main():  # Read the first sector of the first disk as example.
    """Demo usage of function."""
    if os.name == "nt":
        # Windows based OS normally uses '\\.\physicaldriveX' for disk drive identification.
        read_sector(r"\\.\physicaldrive1",0)
        print('done')
    else:
        # Linux based OS normally uses '/dev/diskX' for disk drive identification.
        print(read_sector("/dev/disk0"))


def read_sector(disk, sector_no=0):
    """Read a single sector of the specified disk.
    Keyword arguments:
    disk -- the physical ID of the disk to read.
    sector_no -- the sector number to read (default: 0).
    """
    # Static typed variable
    read = None
    # File operations with `with` syntax. To reduce file handeling efforts.
    with open(disk, 'r+b') as fp:
        # fp.seek(0)
        # for n in range(10):
        #     print(fp.read(1))

        fp.seek(sector_no * 512)
        read = fp.read(4)       # read the signature
        SD_Sig = read.hex()        # 
        print(SD_Sig)
        #print('SD signature : '+ str(SD_Sig))

        read = fp.read(1)       # read file pointer
        SD_FilePtr = int.from_bytes(read, 'little')
        print('Number of files : ' + str(SD_FilePtr))

        read = fp.read(1)       # read isDownloaded
        if int.from_bytes(read, 'little')==1:
            isDownloaded = 'yes'
        else:
            isDownloaded = 'no'

        print('FS already downloaded :' + isDownloaded)

        fileNames = []
        startAddress = []
        fileSizes = []

        for file in range(SD_FilePtr):      # go throuth each file
            read = fp.read(12)      # read file name
            fName = read.decode()
            fileNames.append(fName.rstrip('\x00'))

            read = fp.read(4)
            sAddress = int.from_bytes(read, 'little')
            startAddress.append(sAddress)

            read = fp.read(4)
            ADC_Size = int.from_bytes(read, 'little')

            read = fp.read(4)
            SD_Size = int.from_bytes(read, 'little')
            fileSizes.append(SD_Size)

            print('File name = ' + fName )
            print('start address = ' + str(sAddress))
            print('ADC frames = ' + str(ADC_Size))
            print('SD frames = ' + str(SD_Size))
            print('--------------------------------------------------------------------------------')
        print(fileNames)
        print(startAddress)
        print(fileSizes)
            # fp.seek(sAddress * 512)

        for file in range(SD_FilePtr):
            with open('./'+fileNames[file]+'.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                fp.seek(startAddress[file]*512)
                for i in range(fileSizes[file]):
                    read = fp.read(512)
                    for j in range(0,32):
                        writer.writerow([int.from_bytes(read[j*16:j*16+4], 'little'), read[j*16+4:j*16+8].hex(), read[j*16+8:j*16+12].hex(), read[j*16+12:j*16+16].hex()])

       
    return read


if __name__ == "__main__":
    main()