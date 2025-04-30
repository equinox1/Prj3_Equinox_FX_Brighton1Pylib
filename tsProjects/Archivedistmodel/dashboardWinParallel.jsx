import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Table, TableHeader, TableRow, TableCell } from "@/components/ui/table";
import { RefreshCw } from "lucide-react";

export default function TunerDashboard() {
  const [chiefStatus, setChiefStatus] = useState("Unknown");
  const [workers, setWorkers] = useState([]);

  const fetchStatus = async () => {
    try {
      const response = await fetch("/api/tuner-status"); // <-- Implement backend
      const data = await response.json();
      setChiefStatus(data.chief);
      setWorkers(data.workers);
    } catch (e) {
      console.error("Failed to fetch status", e);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6 grid gap-6 grid-cols-1 md:grid-cols-2">
      <Card>
        <CardContent className="p-4">
          <h2 className="text-xl font-semibold mb-2">Chief Tuner</h2>
          <p className={`font-medium ${chiefStatus === "Online" ? "text-green-600" : "text-red-600"}`}>{chiefStatus}</p>
        </CardContent>
      </Card>

      <Card className="col-span-1 md:col-span-2">
        <CardContent className="p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Worker Status</h2>
            <Button size="sm" onClick={fetchStatus}>
              <RefreshCw className="w-4 h-4 mr-1" /> Refresh
            </Button>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Progress</TableCell>
              </TableRow>
            </TableHeader>
            {workers.map((w) => (
              <TableRow key={w.id}>
                <TableCell>{w.id}</TableCell>
                <TableCell className={w.status === "Running" ? "text-green-600" : "text-red-600"}>{w.status}</TableCell>
                <TableCell><Progress value={w.progress} /></TableCell>
              </TableRow>
            ))}
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}