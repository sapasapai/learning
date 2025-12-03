namespace workforce;


entity Employees  {
  key EmpID        : String(10);
      FirstName    : String(50);
      LastName     : String(50);
      Gender       : String(1);
      Country      : String(3);
      HireDate     : Date;
      CostCenterID : Association to CostCenters;
      BaseSalary   : Decimal(15,2);
      Currency     : String(3);
}

entity CostCenters  {
  key CostCenterID : String(8);
      Name         : String(100);
      Region       : String(10);
      ManagerEmpID : Association to Employees;
}

entity Payroll  {
  key PayrollID   : Integer;
      EmpID       : Association to Employees;
      Period      : String(7);      // YYYY-MM
      GrossAmount : Decimal(15,2);
      TaxAmount   : Decimal(15,2);
      NetAmount   : Decimal(15,2);
      CostCenterID: Association to CostCenters;
      Currency    : String(3);
}