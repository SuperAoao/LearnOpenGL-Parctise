#pragma once
#include <string>
#include <iostream>
#include <vector>

struct Star
{
	float x; // measured in lightyear
	float y;
	float z;
	float absoluteMag;
	Star() : x(0.0),
		y(0.0),
		z(0.0),
		absoluteMag(0.0)
	{

	}
};

class StarDataBase
{
public:
	StarDataBase();
	~StarDataBase();
	bool loadBinaryData(const std::string& path);
	std::vector<float> getVerticsArray(const double d_len, const double t_in);
	unsigned int getStarNum();
private:
	std::vector<Star> m_vecStars;
	unsigned int m_nStars;
};