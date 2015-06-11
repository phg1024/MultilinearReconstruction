#include "mainwindow.h"
#include <QApplication>

#include "multilinearmodelbuilder.h"

#include "test_all.h"

int main(int argc, char *argv[])
{
  testAll();

  MultilinearModelBuilder builder;
  builder.build();
  return 0;

  QApplication a(argc, argv);
  MainWindow w;
  w.show();

  return a.exec();
}
