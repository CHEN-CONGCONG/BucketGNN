#include "Functionalities.h"


void runTest(string str, string whichTest, string &network)
{
	if (str.compare("Debug") == 0)
	{
		if (whichTest.compare("Trunc") == 0)
		{
			network = "Debug Truncate";
			debugTrunc();
		}
		else if (whichTest.compare("MatMul") == 0)
		{
			network = "Debug MatMul";
			debugMatMul();
		}
		else if (whichTest.compare("DotProd") == 0)
		{
			network = "Debug DotProd";
			debugDotProd();
		}
		else if (whichTest.compare("ArrayAccess") == 0)
		{
			network = "Debug ArrayAccess";
			debugArrayAccess();
		}
		else if (whichTest.compare("PC") == 0)
		{
			network = "Debug PC";
			debugPC();
		}
		else if (whichTest.compare("Wrap") == 0)
		{
			network = "Debug Warp";
			debugWrap();
		}
		else if (whichTest.compare("ReLUPrime") == 0)
		{
			network = "Debug ReLUPrime";
			debugReLUPrime();
		}
		else if (whichTest.compare("ReLU") == 0)
		{
			network = "Debug ReLU";
			debugReLU();
		}
		else if (whichTest.compare("SelectShares") == 0)
		{
			network = "Debug SelectShares";
			debugSelectShares();
		}
		else if (whichTest.compare("SelectBitShares") == 0)
		{
			network = "Debug SelectBitShares";
			debugSelectBitShares();
		}
		else if (whichTest.compare("Softmax") == 0)
		{
			network = "Debug Softmax";
			debugSoftmax();
		}
		else if (whichTest.compare("InnerProd") == 0)
		{
			network = "Debug InnerProduct";
			debugInnerProd();
		}
		else if (whichTest.compare("Maxpool") == 0)
		{
			network = "Debug Maxpool";
			debugMaxpool();
		}
		else if (whichTest.compare("Pow") == 0)
		{
			network = "Debug Pow";
			debugPow();
		}
		else if (whichTest.compare("Division") == 0)
		{
			network = "Debug Division";
			debugDivision();
		}
		else if (whichTest.compare("InvSqrt") == 0)
		{
			network = "Debug InvSqrt";
			debugInvSqrt();
		}
		else if (whichTest.compare("Inv") == 0)
		{
			network = "Debug Inv";
			debugInv();
		}
		else if (whichTest.compare("ArrayAccessVector") == 0)
		{
			network = "Debug ArrayAccessVector";
			debugArrayAccessVector();
		} 
		else if (whichTest.compare("BN") == 0)
		{
			network = "Debug BN";
			debugBN();
		} 
		else if (whichTest.compare("ForbeniusNorm") == 0)
		{
			network = "Debug ForbeniusNorm";
			debugForbeniusNorm();
		}
		else if (whichTest.compare("BatchArrayAccess") == 0)
		{
			network = "Debug BatchArrayAccess";
			debugBatchArrayAccess();
		}
		else if (whichTest.compare("BatchArrayAccessVector") == 0)
		{
			network = "Debug BatchArrayAccessVector";
			debugBatchArrayAccessVector();
		}
		else
			assert(false && "Unknown debug mode selected");
		cout << "Debug mode selected, test: ---" << whichTest << "--- completed" << endl;
	}
	else if (str.compare("Test") == 0)
	{	
		if (whichTest.compare("RSS") == 0)
		{
			network = "Test RSS";
            testRSS();
		}
		else
			assert(false && "Unknown test mode selected");
		cout << "Test mode selected, test: ---" << whichTest << "--- completed" << endl;
	}
	else
		assert(false && "Only Debug or Test mode supported");
}