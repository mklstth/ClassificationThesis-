# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os


# Function to rename multiple files
def main():
    i = 0

    for filename in os.listdir("/media/mikes/Data/DATA/mellkas/LIDC-IDRI-0051/01-01-2000-69899/jpg"):
        dst = "has22" + str(i) + ".jpg"
        src = '/media/mikes/Data/DATA/mellkas/LIDC-IDRI-0051/01-01-2000-69899/jpg/' + filename
        dst = '/media/mikes/Data/DATA/mellkas/LIDC-IDRI-0051/01-01-2000-69899/jpg/' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()