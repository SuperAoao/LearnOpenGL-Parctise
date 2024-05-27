#include "5.stardb.h"
#include "5.helper.h"
#include <fstream>

const char* FILE_HEADER = "CELSTARS";
const char* CROSSINDEX_FILE_HEADER = "CELINDEX";

StarDataBase::StarDataBase() : m_nStars(0)
{
}

StarDataBase::~StarDataBase()
{
}

bool StarDataBase::loadBinaryData(const std::string& path)
{
	if (path.empty())
	{
		std::cout << "[Error] Star Database path is empty!" << std::endl;
		return false;
	}
	std::ifstream ifs(path, std::ios::in | std::ios::binary);
	if (!ifs.good())
	{
		std::cout << "failed to open " << path << '\n';
		return false;
	}

	int headerLength = strlen(FILE_HEADER);
	char* header = new char[headerLength];
	ifs.read(header, headerLength);
	if (strncmp(header, FILE_HEADER, headerLength))
	{
		delete[] header;
		return false;
	}
	delete[] header;

	{
		uint16_t version;
		ifs.read((char*)&version, sizeof(version));
		if (version != 0x0100)
			return false;
	}

	unsigned int nStarsInFile = 0;

	ifs.read((char*)&nStarsInFile, sizeof(nStarsInFile));
	if (!ifs.good())
		return false;

	while (((unsigned int)m_nStars) < nStarsInFile)
	{
		uint32_t catNo = 0;
		float x = 0.0f, y = 0.0f, z = 0.0f;
		int16_t absMag;
		uint16_t spectralType;

		ifs.read((char*)&catNo, sizeof catNo);
		ifs.read((char*)&x, sizeof x);
		ifs.read((char*)&y, sizeof y);
		ifs.read((char*)&z, sizeof z);
		ifs.read((char*)&absMag, sizeof absMag);
		ifs.read((char*)&spectralType, sizeof spectralType);
		if (ifs.bad())
			break;

		Star star;
		star.x = x;
		star.y = y;
		star.z = z;
		star.absoluteMag = (float)absMag / 256.0f;
		if (abs(star.absoluteMag - 0.58 ) <= 1e-2)
		{
			double distance = sqrt(star.x * star.x + star.y * star.y + star.z * star.z);
			if (abs (distance - 25) <= 1)
			{
				std::cout << "maybe this is Vega" << std::endl;
			}
			
		}
		m_vecStars.push_back(star);
		/*CFxStarDetails* details = NULL;
		CFxStellarClass sc;
		if (sc.unpack(spectralType))
			details = CFxStarDetails::getStarDetails(sc);

		if (details == NULL)
		{
			return false;
		}

		star.setDetails(details);
		star.setCatalogNumber(catNo);
		unsortedStars.add(star);*/

		m_nStars++;
	}

	if (ifs.bad())
		return false;

	/*if (unsortedStars.size() > 0)
	{
		binFileStarCount = unsortedStars.size();
		binFileCatalogNumberIndex = new CFxStar * [binFileStarCount];
		for (unsigned int i = 0; i < binFileStarCount; i++)
		{
			binFileCatalogNumberIndex[i] = &unsortedStars[i];
		}
		std::sort(binFileCatalogNumberIndex, binFileCatalogNumberIndex + binFileStarCount,
			PtrCatalogNumberOrderingPredicate());
	}*/

	ifs.close();
	return true;
}

std::vector<float> StarDataBase::getVerticsArray(const double d_len, const double t_in)
{
	 std::vector<float> array;
	 array.reserve(m_nStars);
	 for (const Star& star: m_vecStars)
	 {
		 // find vega
		 float vega_x = 3.16450477;
		 float vega_y = 22.2810555;
		 float vega_z = 11.5554714;

		 float x_offset = abs(star.x - vega_x);
		 float y_offset = abs(star.y - vega_y);
		 float z_offset = abs(star.z - vega_z);
		 if (x_offset < 0.1 && y_offset < 0.1 && z_offset < 0.1)
		 {
			 std::cout << "maybe this is vega" << std::endl;
		 }
		 else
		 {
			 //continue;
		 }
		
		 float distance = sqrt(star.x * star.x + star.y * star.y + star.z * star.z);
		 double apparentMag = Celesitial::absToAppMag(star.absoluteMag, distance);
		 if (apparentMag > 6.0)	// skip dark stars for temporary test
		 {
			 //continue;
		 }
		
		 double n = Celesitial::getPhotonsFromApparentMagnitude(apparentMag, d_len, t_in);

		 array.push_back(star.x);
		 array.push_back(star.y);
		 array.push_back(star.z);
		 array.push_back(n);
	 }

	 return array;
}

unsigned int StarDataBase::getStarNum()
{
	return m_nStars;
}
