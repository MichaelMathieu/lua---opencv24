#ifndef __LUA_TEMPLATE_HPP__
#define __LUA_TEMPLATE_HPP__

#ifndef __INSIDE_THPP_HPP__
#error LuaTemplate.hpp should not be included outside of THpp.hpp
#endif

extern "C" {
#include<luaT.h>
}
#include<THpp.hpp>
#include<vector>
#include<string>

template<typename T> inline T FromLuaStack(lua_State* L, int i) {
  THerror("Call of FromLuaStack on a non-implemented type");
  return *((T*)NULL); //avoid warning
}

#define MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(type, lua_fun)	 \
template<> inline type FromLuaStack<type>(lua_State* L, int i) { \
  return (type)lua_fun(L, i);					 \
}
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(bool, lua_toboolean)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(char, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(unsigned char, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(short, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(unsigned short, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(int, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(unsigned int, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(long, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(unsigned long, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(long long, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(unsigned long long, lua_tointeger)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(float, lua_tonumber)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(double, lua_tonumber)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(long double, lua_tonumber)
MAKE_FROM_LUA_STACK_NUMBER_TEMPLATE(std::string, lua_tostring)

#define MAKE_FROM_LUA_STACK_TENSOR_TEMPLATE(type, typestring)		\
  template<> inline TH::Tensor<type> FromLuaStack<TH::Tensor<type> >(lua_State* L, int i) { \
    return TH::Tensor<type>((TH::Types<type>::CTensor*)luaT_checkudata(L, i, luaT_checktypename2id(L, typestring))); \
  }
MAKE_FROM_LUA_STACK_TENSOR_TEMPLATE(float, "torch.FloatTensor")
MAKE_FROM_LUA_STACK_TENSOR_TEMPLATE(double, "torch.DoubleTensor")
MAKE_FROM_LUA_STACK_TENSOR_TEMPLATE(unsigned char, "torch.ByteTensor")
MAKE_FROM_LUA_STACK_TENSOR_TEMPLATE(char, "torch.CharTensor")
MAKE_FROM_LUA_STACK_TENSOR_TEMPLATE(short, "torch.ShortTensor")
MAKE_FROM_LUA_STACK_TENSOR_TEMPLATE(int, "torch.IntTensor")
MAKE_FROM_LUA_STACK_TENSOR_TEMPLATE(long, "torch.LongTensor")

template<typename T> std::vector<T> TableFromLuaStack(lua_State* L, int i) {
  int n = luaL_getn(L, i);
  std::vector<T> ret;
  int newi = (i > 0) ? i : i-1;
  for (int j = 0; j < n; ++j) {
    lua_pushnumber(L, j+1);
    lua_gettable(L, newi);
    ret.push_back(FromLuaStack<T>(L, -1));
  }
  return ret;
}

#define MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(type)			\
  template<>								\
  inline std::vector< type > FromLuaStack<std::vector< type > >(lua_State* L, int i) { \
      return TableFromLuaStack< type >(L, i);				\
    }
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(int)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(long)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(float)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(double)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(std::string)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(TH::Tensor<float>)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(TH::Tensor<double>)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(TH::Tensor<unsigned char>)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(TH::Tensor<char>)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(TH::Tensor<short>)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(TH::Tensor<int>)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(TH::Tensor<long>)
#undef MAKE_FROM_LUA_STACK_TABLE_TEMPLATE


template<typename T> inline void PushOnLuaStack(lua_State* L, const T & topush) {
  THerror("Call of PushOnLuaStack on a non-implemented type");
}
template<> inline void PushOnLuaStack<int>(lua_State* L, const int & topush) {
  lua_pushinteger(L, topush);
}
template<> inline void PushOnLuaStack<long int>(lua_State* L, const long int & topush) {
  lua_pushinteger(L, topush);
}
template<> inline void PushOnLuaStack<float>(lua_State* L, const float & topush) {
  lua_pushnumber(L, topush);
}
template<> inline void PushOnLuaStack<double>(lua_State* L, const double & topush) {
  lua_pushnumber(L, topush);
}

template<typename T> inline T FromLuaStack(int i) {
  return FromLuaStack<T>(L_global, i);
}

template<typename T> inline void PushOnLuaStack(const T & topush) {
  PushOnLuaStack<T>(L_global, topush);
}


#endif
