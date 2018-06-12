/****************************************************************
**	OrangeBot Projects
*****************************************************************
**	      *
**	     *
**	    *  *
**	******* *
**	         *
*****************************************************************
**	Neural Network Example 1
*****************************************************************
**	This examples implement the smallest possible neural network featuring:
**	>one inputù
**	>one output
**	>one neuron
**	The neuron has no activation function and has just one weight and
**	one bias, making it a simple amplifier.
**	Y = W0 *X +W1
**
**	Gain and bias are the weighta. Initialized at random.
**
**	Only two random training patterns are used. Only one line pass through two point.
**	A single exact solution easy to compute traditionally is garanteed to exist
**
**	It introduces the concept of:
**		>Linear Regression: a way to estimate weight corrections from errors
**		>Affinity: a way for correction to be applied to weights that actually work towards the goal
**		>Delta correction:	corrections are computed as variation to current weight value
**		>Learning rate:	a way to control the speed of the learning. Fast learning result in high error or instability
**		>Learning Momentum: a way to pass over local minimum by giving inertia to the correction
**
****************************************************************/

/****************************************************************
**	HISTORY VERSION
****************************************************************
**	V0.1 ALPHA
**		Tried with derivative. unstable
**		Tried with exact single pattern solution. unstable.
**		Problem is I need to handle patterns that want opposite corrections.
**		Tried linear regressions. Works like a charm.
**	V1.0
**		Cleaned up code
****************************************************************/

/****************************************************************
**	KNOWN BUGS
****************************************************************
**
****************************************************************/

/****************************************************************
**	DESCRIPTION
****************************************************************
**
****************************************************************/

/****************************************************************
**	TODO
****************************************************************
**
****************************************************************/

/****************************************************************
**	INCLUDE
****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>	//used for sqrt
#include <time.h>	//used for random

/****************************************************************
**	DEFINES
****************************************************************/

///Training Pattern definitions
#define NUM_PAT_TRAIN		2		//Number of training patterns
#define NUM_BATCH			2		//Training patterns are processed in batches of 1

///Neural network definitions
#define NUM_WEIGHTS			2		//Number of weights. Just the offset and bias of the amplifier

///Learning parameters of the Neural Network
#define LEARN_RATE			0.1		//Learning rate of the neural network.
#define LEARN_MOMENTUM		0.1		//Learning momentum of the neural network
#define LEARN_AFFINITY		0.1		//Minimum learning from a fully decorrelated correction

///Targets
#define MAX_ITER			1000	//Maximum iterations allowed for training
#define ERR_TARGET			1e-4	//Target error

/****************************************************************
**	MACROS
****************************************************************/

//absolute value of input
#define ABS(x)			(((x)>=0)?(x):(-(x)))
//Sign of input
//#define SIGN(x)         (((x)>=0)?(+1.0):(-1.0))

/****************************************************************
**	GLOBAL VARIABILE
****************************************************************/

/****************************************************************
**	FUNCTIONS
****************************************************************/

//Utility: Generate a rendom float in a given range
extern double rand_double( double fmin, double fmax );

/****************************************************************
**	MAIN
****************************************************************
**	INPUT:
**	OUTPUT:
**	RETURN:
**	DESCRIPTION:
****************************************************************/

int main()
{
    ///----------------------------------------------------------------
    ///	STATIC VARIABILE
    ///----------------------------------------------------------------

    ///----------------------------------------------------------------
    ///	LOCAL VARIABILE
    ///----------------------------------------------------------------

    //Fast counter
    register int t;
    //iteration counter
    int iter_cnt;
    //batch handling
    int batch_off, batch_cnt;
    //Continue flag
    uint8_t f_continue;

    ///Training Patterns
    //Arrays of data used to train the neural network
    double xt[NUM_PAT_TRAIN];
    double yt[NUM_PAT_TRAIN];

    ///Weights
    //Weights of the neural network
    //0: Amplifier Gain
    //1: Amplifier Bias
    double w[NUM_WEIGHTS];

    ///Neural Network
    //This network has just an input and an output. no intermediate vars are needed.
    double x, y;

    ///Learning
    //Error
    double err;
    //Overal error over all training and validation patterns
    double total_err;
    //New absolute weight value estimated by linear regression
    double new_w[NUM_WEIGHTS];
    //delta correction to weights
    double dw[NUM_WEIGHTS];
    //Previous effective delta correction.
    double old_dw[NUM_WEIGHTS];

    ///Linear Regression Learning
    //Desired output of the element being trained
    double yw;
    //Xavg = average of inputs of the amplifier. Accumulated over each training input.
    double x_avg;
    //Yavg = average of the desired outputs of the amplifier. Accumulated over each training input.
    double y_avg;
    //XYavg = average of the product of input and desired output of the amplifier. Accumulated over each training input.
    double xy_avg;
    //X2avg = average of squares of input of amplifier. Accumulated over each training input.
    double x2_avg;

    ///----------------------------------------------------------------
    ///	CHECK AND INITIALIZATIONS
    ///----------------------------------------------------------------

    //Initialize random number generator
    srand(time(NULL));

    //f = fopen( "learn.log", "w+" );

    ///----------------------------------------------------------------
    ///	BODY
    ///----------------------------------------------------------------

    printf("Neural Network Example 1\n");
    printf("Just a single amplifier with bias and offset.\n");
    printf("Trained using Neural Network gradient descent algorithm\n");

    ///----------------------------------------------------------------
    ///	STAGE 1: Generate Training Patterns
    ///----------------------------------------------------------------

    printf("----------------------------------------------------------------\n");
    printf("Define two points. Only one valid line exist.\n\n");
    printf("Training patterns:\n");

    //For: every pattern
    for (t = 0; t < NUM_PAT_TRAIN; t++)
    {
        //Generate random point
        xt[t] = rand_double( -1.0, +1.0 );
        yt[t] = rand_double( -1.0, +1.0 );

        printf("pat%-2d | X: %+-1.3f | y: %+-1.3f\n", t, xt[t], yt[t] );
    }

    ///----------------------------------------------------------------
    ///	STAGE 2: Initial Seeding
    ///----------------------------------------------------------------
    // For the algorithm to work, weights must be initialized to something

    //For every weight
    for (t = 0; t < NUM_WEIGHTS; t++)
    {
        //Initialize the weight to random
        w[t] = rand_double( -1.0, +1.0 );
    }

    ///----------------------------------------------------------------
    ///	STAGE 3: Training
    ///----------------------------------------------------------------
    //	It involves several phases
    //	All training patterns are processed and the error calculated
    //	The error is used to compute a delta that when applied brings the error closer to zero
    //	Each training pattern has an idea on what weights to modifying
    //	By averaging all corrections for all patterns, the overall correction is found
    //	Loop over and over until the overall error get below the threshold

    //For all weights
    for (t = 0; t < NUM_WEIGHTS; t++)
    {
        //Initialize memory of learning array. Used for learning momentum.
        old_dw[t] = 0;
    }

    //Initialize
    f_continue = 1;
    batch_off = 0;
    batch_cnt = 0;
    //While: I'm not done learning
    while (f_continue == 1)
    {
        ///----------------------------------------------------------------
        ///	PROCESS TRAINING PATTERNS
        ///----------------------------------------------------------------
        //	Train one itertion of the Neural Network using a batch of training patterns

        ///Initializations
        //Initialize error
        total_err = 0.0;
        //Linear Regression accumulators
        x_avg = 0.0;
        y_avg = 0.0;
        xy_avg = 0.0;
        x2_avg = 0.0;
        //Process the current batch of training data
        t = batch_off;
        while ((t < NUM_PAT_TRAIN) && (batch_cnt < NUM_BATCH))
        {
            ///----------------------------------------------------------------
            ///	EXECUTE
            ///----------------------------------------------------------------
            //Execute Neural Network on given training pattern

            ///Input
            //Fetch input
            x = xt[t];

            ///Amplifier
            //	Out = In *Gain +Bias
            //Compute output of the amplifier
            y = x *w[0] +w[1];

            ///----------------------------------------------------------------
            ///	ERROR
            ///----------------------------------------------------------------

            //Compute error of the current network on training pattern
            err = y -yt[t];

            printf("x: %+-1.3f | y: %+-1.3f | err: %+-1.3e\n", x, y, err);

            //accumulate the current square error
            total_err += err*err;

            ///----------------------------------------------------------------
            ///	LINEAR REGRESSION ACCUMULATION
            ///----------------------------------------------------------------
            //	In the end, what i want to do is:
            //	Given an input pattern, an actual output pattern and an error pattern
            //	compute the parameters of the amplifier that minimize the square sum of error
            //
            //	The idea is to propagate the error backward, use the error to compute the desired output
            //	then use a linear regression algorithm to compute the ideal values of gain and bias
            //
            //	G' = (XYavg -Xavg *Yavg) / (X2avg -Xavg *Xavg)
            //	B' = Yavg -G'*Xavg

            ///	Compute desired output
            //	In this example is trivial, it's the training output.
            //	In a general example ywish = y -err
            yw = y -err;

            ///	Accumulate averages for this element
            //
            x_avg += x;
            y_avg += yw;
            xy_avg += x *yw;
            x2_avg += x *x;

            ///----------------------------------------------------------------
            ///	NEXT PATTERN
            ///----------------------------------------------------------------

            batch_cnt++;
            t++;
        }	//End For: every training pattern

        ///----------------------------------------------------------------
        ///	NEXT BATCH
        ///----------------------------------------------------------------

        //If all patterns have been used, restart
        if (t>=NUM_PAT_TRAIN)
        {
            //start again from first pattern
            t = 0;
            //TODO: shuffle patterns between batches.
        }
        //if: all batch has been used
        if (batch_cnt >= NUM_BATCH)
        {
            //Start batch again from zero
            batch_cnt = 0;
            //Count from next unprocessed pattern
            batch_off = t;
            //works even if NUM_PAT_TRAIN is not an integer multiple of NUM_BATCH
        }

        ///----------------------------------------------------------------
        ///	COMPUTE ERROR
        ///----------------------------------------------------------------
        //	Overall square error of the neural network
        //	TODO compute error for validation patterns as well.

        total_err = total_err /NUM_BATCH;
        total_err = sqrt( total_err );

        ///----------------------------------------------------------------
        ///	LINEAR REGRESSION
        ///----------------------------------------------------------------
        //	Calculate optimal G, B values with linear regression
        //	g' = new gain
        //	b' = new bias
        //	There is a lot of math behind this step.
        //	It's just like a trend line in excel in concept
        //	I have a cloud of points and I want to find the slope and offset
        //	of a line that fits them best.

        ///Calculate averages
        //Those are the four accumulated parameters that allow to calculate the optimal G, B
        x_avg /= NUM_BATCH;
        y_avg /= NUM_BATCH;
        xy_avg /= NUM_BATCH;
        x2_avg /= NUM_BATCH;

        //optimal gain. Optimazed as minimal square distance between linearized output and output data.
        new_w[0] = (xy_avg -x_avg *y_avg) / (x2_avg -x_avg *x_avg);
        //optimal bias
        new_w[1] = y_avg -new_w[0] *x_avg;

        ///----------------------------------------------------------------
        ///	APPLY LEARNING CORRECTIONS
        ///----------------------------------------------------------------
        //	LEARNING RATE
        //	Current gain and bias have to move toward new gain and bias.
        //	Learning rate controls the speed of this transition
        //	w = (1-L)*w +L*w'
        //	b = (1-L)*b +L*b'
        //
        //	LEARNING MOMENTUM
        //	I add inertia to the learning. It's like considering previous knowledge
        //	Allows to pass over local minimums
        //
        //	AFFINITY
        //	I compute the correlation between the input and the output patterns
        //	in this node.
        //	High correlation means that the new weights are correct.
        //	Low correlation means that this weight has nothing to do with the output.
        //	This is a very elegant way to lower the weights that don't help at any time
        //	w = c*(1-L)*w +c*L*w'
        //	b = c*(1-L)*b +c*L*b'

        printf("    g: %+-1.3f | b: %+-1.3f | ", w[0], w[1]);

        //For all weights
        for (t = 0; t < NUM_WEIGHTS; t++)
        {
            //Calculate current effective delta correction to the weight
            dw[t] = LEARN_MOMENTUM *old_dw[t] + (1.0-LEARN_MOMENTUM) *LEARN_RATE *(new_w[t] -w[t]);
            //Save current delta correction
            old_dw[t] = dw[t];
            //Apply correction
            w[t] += dw[t];
        }	//End For all weights

        printf("dg: %+-1.3e | db: %+-1.3e\n", dw[0], dw[1]);

        ///----------------------------------------------------------------
        ///	STOP CONDITION
        ///----------------------------------------------------------------
        //	When iteration budget is used up
        //	When error is below threshold
        //	When error reaches maturity

        if ((iter_cnt > MAX_ITER) || (total_err < ERR_TARGET))
        {
            f_continue = 0;
        }
        //Increase overall iteration counter
        iter_cnt++;
    }	//End While: I'm not done learning

    printf("Total iterations: %d\n", iter_cnt);
    printf("Final error: %+-1.3e\n", total_err);

    ///----------------------------------------------------------------
    ///	FINALIZATIONS
    ///----------------------------------------------------------------

    return 0;
}	//end function: main

/****************************************************************************
**	rand_double | double, double
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
**	Generate a random double
****************************************************************************/

double rand_double( double fmin, double fmax )
{
    ///--------------------------------------------------------------------------
    ///	STATIC VARIABILE
    ///--------------------------------------------------------------------------

    ///--------------------------------------------------------------------------
    ///	LOCAL VARIABILE
    ///--------------------------------------------------------------------------

    //int random
    int irnd;
    //float random
    double frnd;

    ///--------------------------------------------------------------------------
    ///	CHECK AND INITIALIZATIONS
    ///--------------------------------------------------------------------------

    ///--------------------------------------------------------------------------
    ///	BODY
    ///--------------------------------------------------------------------------

    //Get a random number
    irnd = rand();
    //Convert directly into float.
    frnd = fmin +(fmax -fmin)*(1.0*irnd)/(1.0*RAND_MAX);

    ///--------------------------------------------------------------------------
    ///	RETURN
    ///--------------------------------------------------------------------------

    return frnd;
}	//end function: rand_double | double, double
