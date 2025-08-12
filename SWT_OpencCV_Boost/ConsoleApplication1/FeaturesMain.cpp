#define _CRT_SECURE_NO_DEPRECATE
//#define _HAS_ITERATOR_DEBUGGING 0
//#define _CRT_SECURE_NO_WARNINGS
#include <cassert>
#include <fstream>
#include "TextDetection.h"
#include <opencv/highgui.h>
#include <exception>

void convertToFloatImage ( IplImage * byteImage, IplImage * floatImage )
{
  cvConvertScale ( byteImage, floatImage, 1 / 255., 0 );
}

class FeatureError : public std::exception
{
std::string message;
public:
FeatureError ( const std::string & msg, const std::string & file )
{
  std::stringstream ss;

  ss << msg << " " << file;
  message = msg.c_str ();
}
~FeatureError () throw ( )
{
}
};

IplImage * loadByteImage ( const char * name )
{
  IplImage * image = cvLoadImage ( name );

  if ( !image )
  {
    return 0;
  }
  cvCvtColor ( image, image, CV_BGR2RGB );
  return image;
}

IplImage * loadFloatImage ( const char * name )
{
  IplImage * image = cvLoadImage ( name );

  if ( !image )
  {
    return 0;
  }
  cvCvtColor ( image, image, CV_BGR2RGB );
  IplImage * floatingImage = cvCreateImage ( cvGetSize ( image ),
                                             IPL_DEPTH_32F, 3 );
  cvConvertScale ( image, floatingImage, 1 / 255., 0 );
  cvReleaseImage ( &image );
  return floatingImage;
}

int mainTextDetection ( int argc, char * * argv )
{
  IplImage * byteQueryImage = loadByteImage ( argv[1] );
  if ( !byteQueryImage )
  {
    printf ( "couldn't load query image\n" );
    return -1;
  }

  // Detect text in the image
  IplImage * output = textDetection ( byteQueryImage, atoi(argv[3]) );
  cvReleaseImage ( &byteQueryImage );
  //int cvSaveImage( const char* filename, const CvArr* image ); save image to file
  cvSaveImage ( argv[2], output );
  cvReleaseImage ( &output );
  return 0;
}
/*main(int argc, char *argv[])
The name of the program is one of the elements of *argv[] and argc is the count of the number of arguments in *argv[]
The program name counts as the first argument.argv[0] will be a string containing the program's name or a null string if that is not available. Remaining elements of argv represent the arguments supplied to the program.
$ gcc mysort.c -o mysort

$ ./mysort 2 8 9 1 4 5
argc = 7
argv[ 0 ] = ./mysort
argv[ 1 ] = 2
argv[ 2 ] = 8
argv[ 3 ] = 9
argv[ 4 ] = 1
argv[ 5 ] = 4
argv[ 6 ] = 5
===================================
int main( int argc, char* argv[] )
  {
  cout << "The name used to start the program: " << argv[ 0 ]
       << "\nArguments are:\n";
  for (int n = 1; n < argc; n++)
    cout << setw( 2 ) << n << ": " << argv[ n ] << '\n';
  return 0;
  }
  
 From command prompt>>>
 D:\prog\test> a Hello world!
The name used to start the program: a
Arguments are:
1: Hello
2: world!

D:\prog\test> cd ..

D:\prog> test\a.exe "Peter Piper" picked a peck of "pickled peppers"
The name used to start the program: test\a.exe
Arguments are:
1: Peter Piper
2: picked
3: a
4: peck
5: of
6: pickled peppers
==========================================
 std::cout << "Have " << argc << " arguments:" << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << std::endl;
		Running it with ./test a1 b2 c3 will output

Have 4 arguments:
./test
a1
b2
c3

Argv[0] is always the (full path)/executableName as a nul terminated string
===============================================
if(argc!=4)
{cout<<no. of arg passed is not equal to 3; return 1}
else
{
argv[0]:program name;argv[1]=1st argument;argv[2]=2nd argument;argv[3]=3rd argument;
return 0;
}
*/

int main ( int argc, char * * argv )
{
  if ( ( argc != 4 ) )
  {
    printf ( "usage: %s imagefile resultImage darkText\n",
             argv[0] );

    return -1;
  }
  return mainTextDetection ( argc, argv );
}