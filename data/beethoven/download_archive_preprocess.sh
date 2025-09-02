# conda activate py2 # preprocess.py and _2npy.py is python2

# Requires 2GB of free disk space at most.
SCRIPTPATH=$( cd "$(dirname "$0")" ; pwd -P )
DL_PATH="$SCRIPTPATH"/download/
mkdir -p "$DL_PATH"
echo "Downloading files to "$DL_PATH""
# See: https://blog.archive.org/2012/04/26/downloading-in-bulk-using-wget/
wget -r -H -nc -nH --cut-dir=1 -A .ogg -R *_vbr.mp3 -e robots=off -P "$DL_PATH" -l1 -i ./itemlist.txt -B 'http://archive.org/download/'
echo "Organizing files and folders"
mv "$DL_PATH"*/*.ogg "$DL_PATH"
rm -rf "$DL_PATH"*/
echo "Converting from OGG to 48Khz, 16bit mono-channel WAV"
# Next line with & executes in a forked shell in the background. That's parallel and not recommended.
# Remove if causing problem
#for file in "$DL_PATH"*_64kb.mp3; do ffmpeg -i "$file" -ar 16000 -ac 1 "$DL_PATH""`basename "$file" _64kb.mp3`.wav" & done 
for file in "$DL_PATH"*.ogg; do
	ffmpeg -i "$file" -ar 48000 -ac 1 "$DL_PATH""`basename "$file" .ogg`.wav"
done 
echo "Cleaning up"
rm "$DL_PATH"*.ogg

# Comment out the following because it convert the .wav file into .flac (compressed), i.e. chunks of 8s.

#echo "Preprocessing"
#python preprocess.py "$DL_PATH" # split into 8s chunks, Note: set fs = 48k
#echo "Done!"

#echo "Writing datasets"
#python _2npy.py # write into train, test sets of py files
#echo "Done!"
