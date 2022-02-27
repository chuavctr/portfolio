// DSA Assignment.cpp 
// Victor Chua Min Chun 0129219
// UOW Malaysia KDU
//

#include <iostream>
//#include <cstdlib> 
#include <string>
#include <chrono>
#include <ctime>

using namespace std;

//Structure Declarations
struct depend //Dependant info
{
	string dName, relation;
	long long dicNum;
};

struct user //User info
{
	string uName, address, status, redZ, uhpNum;
	long long uicNum;
};

struct history
{
	int depCount;
	string sec, min, hr, day, mon, yr, apm;
	string location, address, deps;
	history* nextH;
};

struct history* hTop;

/*
Title: std::to_string
Author: cplusplus.com
Date: 28-3-2019
Availability: https://www.cplusplus.com/reference/string/to_string/
*/
string zeros(int value) {
	string add;
	if (value < 10) {
		add = "0" + to_string(value);
	}
	else {
		add = to_string(value);
	}
	return add;
}

string locCheck() {
	history* hist = new history();
	string redZone[4] = { "Pavillion", "Restoran Al-Maju", "Bok Choy Restaurant", "Warung Encik Afiq" };
	string redZ = "";
	hist = hTop;
	while (hist != NULL)
	{
		for (int i = 0; i < 4; i++) {
			if (hist->location == redZone[i]) {
				redZ = "Alert      : This user has been in vicinity of a red zone";	
			}
		}
		hist = hist->nextH;
	}
	return redZ;
}

// Utility function to add an element
// data in the stack insert at the beginning
// To add: Hardcoded locations to change user status
void addLoc(string loc,int depC, string depL)
{
	time_t now;
	struct tm nowLocal;
	now = time(NULL);
	nowLocal = *localtime(&now);
	history* hist = new history();
	int input;
	string sec, min, hr, day, mon, yr;

	if (!hist)
	{
		cout << "\nWarning; database is full!"; //Overflow check
		return;
	}

	sec = zeros(nowLocal.tm_sec);
	min = zeros(nowLocal.tm_min); 
	hr = zeros(nowLocal.tm_hour);
	day = zeros(nowLocal.tm_mday);
	mon = zeros(nowLocal.tm_mon + 1);
	yr = nowLocal.tm_year + 1900;
	hist->location = loc;
	hist->deps = depL;
	hist->sec = sec;
	hist->min = min; 
	hist->hr = hr;
	hist->day = day;
	hist->mon = mon;
	hist->yr = yr;
	hist->nextH = hTop;
	hist->depCount = depC;
	if (nowLocal.tm_hour < 12) {
		hist->apm = "AM";
	}
	else {
		hist->apm = "PM";
	}
	hTop = hist;
	return;
}

void addLoc(string loc)
{
	time_t now;
	struct tm nowLocal;
	now = time(NULL);
	nowLocal = *localtime(&now);
	history* hist = new history();
	int input, depC = 0;
	string sec, min, hr, day, mon, yr;

	if (!hist)
	{
		cout << "\nWarning; database is full!"; //Overflow check
		return;
	}

	sec = zeros(nowLocal.tm_sec);
	min = zeros(nowLocal.tm_min);
	hr = zeros(nowLocal.tm_hour);
	day = zeros(nowLocal.tm_mday);
	mon = zeros(nowLocal.tm_mon + 1);
	yr = nowLocal.tm_year + 1900;
	hist->location = loc;
	hist->sec = sec;
	hist->min = min;
	hist->hr = hr;
	hist->day = day;
	hist->mon = mon;
	hist->yr = yr;
	hist->nextH = hTop;
	if (nowLocal.tm_hour < 12) {
		hist->apm = "AM";
	}
	else {
		hist->apm = "PM";
	}
	hTop = hist;
	return;
}

// Check if stack is empty
int isEmpty()
{
	return hTop == NULL;
}

/*
Title: C++ Program to implement stack
Author: Nitya Raut
Date: 28-3-2019
Availability: https://www.tutorialspoint.com/cplusplus-program-to-implement-stack
*/
// Utility function to return top element in a stack
string peek()
{
	// Check for empty stack
	if (!isEmpty())
	{
		cout << "Last visited: " << hTop->location << endl;
		return hTop->location;
	}
	else
	{
		cout << "Last visited: None" << endl;
		return "";
	}
}


//Display location list
void tHist(int dNum)
{
	history* hist = new history();
	// Check for stack underflow
	if (hTop == NULL)
	{
		cout << "\nThere are currently no visits! \n";
		//exit(1);
		return;
	}
	else
	{
		hist = hTop;
		while (hist != NULL)
		{

			if (dNum == 0) {
				cout << "\n===============\n" << hist->location << "\n" << hist->hr << ":" << hist->min << ":" << hist->sec << hist->apm  << "\n===============" << endl;
			}
			else if (dNum > 0) {
				if (hist->deps == "") {
					hist->deps = "None";
				}
				cout << "\n===============\n" << hist->location << "\n" << hist->hr << ":" << hist->min << ":" << hist->sec << hist->apm << "\nDependants: " << hist->deps << "\n===============" << endl;
			}
			hist = hist->nextH;
		}
		
	}
}

int rAssess(string &con) { //Health Risk Assessment
	int select, rLevel = 0;
	con = "";
	system("CLS");
	cout << "======================Part 1: Health Assessment======================\n1. Are you exhibiting any of the following symptoms? \n- Fever\n- Sore Throat\n- Cough\n- Shortness of Breath\n\n1. No\n2. Yes\n" << endl;
	cin >> select;
	cin.ignore();
	if (select == 2) {
		rLevel = 1;
		con = "- Exhibiting one or more COVID-19 symptoms\n";
	}

	system("CLS");
	cout << "======================Part 1: Health Assessment======================\n2. Do you have any previous health complications, such as: \n- Asthma\n- Diabetes\n- Hypertension\n- Heart Disease\n- Cancer\n\n1. No\n2. Yes\n" << endl;
	cin.clear();
	cin >> select;
	cin.ignore();
	if (select == 2) {
		rLevel = 2;
		if (con == "") {
			con = "- Have one or more existing health complications\n";
		}
		else {
			con = con + "- Have one or more existing health complications\n";
		}
	}

	system("CLS");
	cout << "======================Part 1: Health Assessment======================\n3. Have you travelled or resided in any country outside Malaysia in the last 14 days?\n\n1. No\n2. Yes\n" << endl;
	cin.clear();
	cin >> select;
	cin.ignore();
	if (select == 2) {
		rLevel = 3;
		if (con == "") {
			con = "- Travelled abroad in the last 14 days\n";
		}
		else {
			con = con + "- Travelled abroad in the last 14 days\n";
		}
	}

	system("CLS");
	cout << "======================Part 1: Health Assessment======================\n4. Have you had close contact with a COVID-19 patient in the last 14 days?\n\n1. No\n2. Yes\n" << endl;
	cin.clear();
	cin >> select;
	cin.ignore();
	if (select == 2) {
		rLevel =4;
		if (con == "") {
			con = "- In close proximity with a COVID-19 positive individual\n";
		}
		else {
			con = con + "- In close proximity with a COVID-19 positive individual\n";
		}
	}

	system("CLS");
	cout << "======================Part 1: Health Assessment======================\n5. Have you been to COVID-19 affected areas?\n\n1. No\n2. Yes\n" << endl;
	cin.clear();
	cin >> select;
	cin.ignore();
	if (select == 2) {
		rLevel = 5;
		if (con == "") {
			con = "- Been in vicinity of a designated COVID-19 red zone\n";
		}
		else {
			con = con + "- Been in vicinity of a designated COVID-19 red zone\n";
		}
	}
	return rLevel;
}

int main()
{
	user* u = new user();
	depend dep[8];
	int menu, select, depNum, depC, rStat = 0;
	string depList, condition;
	char location[100];

	// User Registration
	cout << "======================Welcome to MySalam====================== \nPlease enter your name:" << endl;
	getline(cin, u->uName);
	cout << "\nPlease enter your Idenfication Card (IC) No.: " << endl;
	cin >> u->uicNum;
	cin.ignore();
	cin.clear();
	cout << "\nPlease enter your H/P No. :" << endl;
	cin >> u->uhpNum;
	cin.ignore();
	cin.clear();
	cout << "\nPlease enter your address: " << endl;
	getline(cin, u->address);

	// Health assessment
	rStat = rAssess(condition); 

	//Dependant Registration
	system("CLS");
	cout << "======================Part 2: Dependants====================== \nDo you have any dependants? \n1 - yes\n2 - No" << endl;
	cin >> select;
	cin.ignore();
	if (select == 1) {
		system("CLS");
		cout << "======================Part 2: Dependants====================== \nHow many dependants? \n" << endl;
		cin >> depNum;
		cin.ignore();
		if (depNum != 0) {
			for (int i = 0; i < depNum; i++) {
				system("CLS");
				cout << "======================Part 2: Dependants====================== \n" << endl;
				cout << "Please enter dependant " << i + 1 << "'s name: \n" << endl;
				getline(cin, dep[i].dName);
				cout << "Please enter " << dep[i].dName << "'s Idenfication Card(IC) No.: \n" << endl;
				cin >> dep[i].dicNum;
				cin.ignore();
				cout << "What is " << dep[i].dName << "'s relationship to you?\n" << endl;
				getline(cin, dep[i].relation);
			}
			system("CLS");
			cout << "======================Part 2: Dependants====================== \nDependant List: " << endl;
			for (int i = 0; i < depNum; i++)
			{
				cout << "\n==========Dependant " << i + 1 << "==========" << endl;
				cout << "Name     : " << dep[i].dName << endl;
				cout << "Relation : " << dep[i].relation << endl;
				cout << "IC No.   : " << dep[i].dicNum << "\n" << endl;
			}	
		}
		else {
			cout << "======================Part 2: Dependants====================== \nYou have entered 0 dependants. Redirecting to menu..." << endl;
		}
		system("pause");
	}
	else if(select ==2 ) {
		depNum = 0;
		cout << "Loading..." << endl;
	}
	else {
		depNum = 0;
		cout << "Error: Invalid Input \nMenu Shutting down..." << endl;
		system("pause");
	}

	//Main Menu
	
	do {
		system("CLS");
		cout << "======================MySalam======================" << endl;
		cout << "Good day, " << u->uName << endl;
		cout << "ID No       : " << u->uicNum << endl;
		peek();
		cout << "===================================================" << endl;
		cout << "What would you like to do today? \n 1. Register Location \n 2. Travel History \n 3. User Profile \n [Press any other key to exit]" << endl; 
		cin >> menu;
		cin.ignore();

		switch (menu) {
		case 1:// Location Registration
		{
			system("CLS");
			cout << "Register Location" << endl;
			cout << "Please enter venue name: " << endl;
			cin.clear();
			cin.getline(location, sizeof(location));
			//getline(cin, location);
			//cin.ignore();
			//If dependant exist, ask to register dependants
			if (depNum > 0) {
				cout << "Register dependants? \n1. Yes \n2. No (Press any other key)" << endl;
				cin >> select;
				cin.ignore();
				if (select == 1) {
					system("CLS");
					depC = 0;
					do {
						for (int i = 0; i < depNum; i++)
						{
							cout <<  i + 1 << ". " << dep[i].dName << " (" << dep[i].relation << ")" << endl;
						}
						cout << "Select dependant: \n" << endl;
						cin.clear();
						cin >> select;
						cin.ignore();
						if (depC == 0) { //Dependant history save formatting. If > 1 name, variable value is saved as "[Name], [Name]".
							depList = dep[select - 1].dName;
						}
						else if (depC > 0) {
							depList = depList + ", " + dep[select - 1].dName;
						}
						if (depC + 1 == depNum) {
							break;
						}
						cout << "Dependants " << depList << " added. \nWould you like to add dependants?\n1. Yes 2.No" << endl;
						cin >> select;
						cin.ignore();
						depC++;
					} while (select != 2);
					cout << "Dependant(s) registered: " << depList << endl;
					system("pause");
					addLoc(location, depC, depList);
				}
				else {
					addLoc(location);
				}
			}
			else {
				addLoc(location);
			}
			//Cross check with COVID-19 affected areas
			u->redZ = locCheck();
			break;
		}
			
		case 2: // List travel history
		{
			system("CLS");
			cout << "Travel History" << endl;
			tHist(depNum);
			system("pause");
			break;
		}
		case 3: // Display user profile
		{
			string status, sDesc;
			switch (rStat) {
				case 1: {
					status = "Person Under Surveillance (PUS)";
					sDesc = "This user is under surveillance for exhibiting one or more symptoms of COVID-19.";
					break;
				}

				case 2: {
					status = "Immmunocompromised";
					sDesc = "This user has existing health complications.";
					break;
				}

				case 3: {
					status = "Person Under Surveillance (PUS)";
					sDesc = "This user is under surveillance after returning from Abroad for 14 days.";
					break;
				}
				case 4: {
					status = "Close Contact";
					sDesc = "This user has come into close proximity with a COVID-19 patient.";
					break;
				}
				case 5: {
					status = "Casual Contact";
					sDesc = "This user has been in an area classified as a red zone.";
					break;
				}
				default: {
					status = "Low Risk";
					sDesc = "This user is not in any immediate risk of COVID-19.";
				}
			}

			system("CLS");
			cout << "====================================================================================================================\n--User Profile------------------------------------------------------------------------------------------------------\n====================================================================================================================\n" << endl;
			cout << "Name       : " << u->uName << "\tStatus     : " << status << "\n" << sDesc << endl;
			cout << "IC No.     : " << u->uicNum << endl;
			cout << u->redZ << endl;
			cout << "H/P No.    : " << u->uhpNum << endl;
			cout << "Address    : " << u->address << endl;
			cout << "Conditions : \n" << condition << endl;
			if (depNum > 0) {
				cout << "\n==========Dependants==================== " << endl;
				for (int i = 0; i < depNum; i++)
				{
					cout << "\n====================" << endl;
					cout << "Name     : " << dep[i].dName << endl;
					cout << "Relation : " << dep[i].relation << endl;
					cout << "IC No.   : " << dep[i].dicNum << endl;
					cout << "====================\n" << endl;
				}
			}
			system("pause");
			break;
		}
		default:
		{
			cout << "System shutting down..." << endl;
			system("pause");
			exit(1);
			break;
		}
		}
	} while (menu != 4);

	return 0;
}

