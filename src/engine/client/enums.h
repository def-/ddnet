/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef ENGINE_CLIENT_ENUMS_H
#define ENGINE_CLIENT_ENUMS_H

enum
{
	NUM_DUMMIES = 2,
	/**
	 * Maximum number of additional servers that can be observed at the same time,
	 * on top of the server the client is connected to.
	 */
	MAX_OBSERVERS = 8,
};

#endif
